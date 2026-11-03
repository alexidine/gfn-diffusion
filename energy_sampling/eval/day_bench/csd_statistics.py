"""Bare CSD frequencies for allocating Day-benchmark draws.

Streams the local Cambridge Structural Database once and counts, for two
populations of Z' = 1 organic homomolecular crystals (C/H/N/O/F only, and
any organic chemistry), the space-group frequency by IT number, the share and
renormalised frequency of Day's 26 space groups, the conventional Z and the
molecules-per-primitive-cell distributions.  Raw counts and counts
deduplicated by refcode family (first six letters; first passing member in
database order) are both reported.  No molecular-weight, rotatable-bond or
other Day-shaped filter is applied.

Needs the CSD Python API (ccdc) and a licensed local CSD; it does not import
anything from this repository.  Run from the CSD venv, e.g.

    set CUDA_VISIBLE_DEVICES=-1
    <csd venv>/python.exe csd_statistics.py --scratch <dir> --out csd_statistics.json
    <csd venv>/python.exe csd_statistics.py --check     # known-refcode check only

The full pass checkpoints its counters to ``--scratch`` every
``--checkpoint_every`` entries and resumes from the checkpoint if one exists.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import json
import os
import re
import time

DAY_26 = (1, 2, 4, 5, 7, 9, 13, 14, 15, 18, 19, 20, 29, 33, 43, 56, 60, 61,
          76, 86, 88, 96, 145, 148, 154, 169)
CHNOF = frozenset('C H N O F'.split())
# number of lattice-centring translations of the conventional cell, keyed by
# ccdc's Crystal.lattice_centring string
CENTRING = {'primitive': 1, 'A-centred': 2, 'B-centred': 2, 'C-centred': 2,
            'I-centred': 2, 'F-centred': 4, 'R-obverse': 3, 'R-reverse': 3}
POPULATIONS = ('CHNOF_zp1', 'organic_zp1')
_ELEMENT_TOKEN = re.compile(r'^([A-Z][a-z]?)(\d*)$')
_MULTIPLIER = re.compile(r'^\d+\((.*)\)$')

FILTERS = [
    "entry.is_organic is True (CSD organic flag: no metal atoms)",
    "entry.is_polymeric is False",
    "entry.has_3d_structure is True",
    "entry.has_disorder is False (CSD disorder flag)",
    "homomolecular: entry.formula has exactly one comma-separated component "
    "(no solvent, counter-ion or co-former) and carries no charge token",
    "entry.crystal.z_prime == 1 exactly",
    "the asymmetric-unit molecule has exactly one connected component "
    "(excludes Z'=1 entries built from two half-molecules on special "
    "positions, and entries with broken connectivity); excluded counts in "
    "excluded_two_or_more_asu_components",
    "conventional Z equals the number of symmetry operators (one molecule on a "
    "general position); violations counted in excluded_z_ne_nops",
    "CHNOF_zp1 only: every element of entry.formula is in {C, H, N, O, F}",
]


def parse_formula(formula):
    """Return the element set of a single-component formula, or None.

    None means the entry is not a single neutral chemical component: more
    than one comma-separated part, a charge token, or an unparsable token.
    """
    parts = [p.strip() for p in formula.split(',') if p.strip()]
    if len(parts) != 1:
        return None
    body = parts[0]
    m = _MULTIPLIER.match(body)
    if m:
        body = m.group(1)
    elements = set()
    for tok in body.split():
        mt = _ELEMENT_TOKEN.match(tok)
        if mt is None:
            return None
        elements.add(mt.group(1))
    return elements or None


def describe(entry):
    """Per-entry crystallographic quantities used by the counts."""
    c = entry.crystal
    sg_number, sg_setting = c.spacegroup_number_and_setting
    n_ops = len(c.symmetry_operators)
    centring = CENTRING.get(c.lattice_centring)
    z = int(round(c.z_value)) if c.z_value else None
    return {
        'refcode': entry.identifier,
        'sg_number': int(sg_number) if sg_number else None,
        'sg_setting': sg_setting,
        'sg_symbol': c.spacegroup_symbol,
        'z': z,
        'z_prime': c.z_prime,
        'n_ops': n_ops,
        'lattice_centring': c.lattice_centring,
        'n_centring': centring,
        'z_primitive': (z // centring) if (z and centring and z % centring == 0) else None,
    }


def new_population():
    return {'raw': collections.Counter(), 'dedup': collections.Counter(),
            'families': set(), 'settings': collections.Counter()}


def new_state():
    return {'n_scanned': 0, 'pops': {p: new_population() for p in POPULATIONS},
            'funnel': collections.Counter()}


def add(pop, key, family, first_in_family):
    pop['raw'][key] += 1
    if first_in_family:
        pop['dedup'][key] += 1


def classify(entry, state):
    """Apply the filters to one entry and update the counters."""
    f = state['funnel']
    f['scanned'] += 1
    if not entry.is_organic:
        return
    f['organic'] += 1
    if entry.is_polymeric:
        return
    f['organic_nonpolymeric'] += 1
    if not entry.has_3d_structure:
        return
    f['has_3d'] += 1
    if entry.has_disorder:
        return
    f['no_disorder'] += 1
    elements = parse_formula(entry.formula)
    if elements is None:
        return
    f['homomolecular'] += 1
    try:
        d = describe(entry)
    except Exception:  # noqa: BLE001 -- entry with no usable crystal
        f['crystal_unreadable'] += 1
        return
    if d['z_prime'] is None or d['z_prime'] != 1:
        return
    f['zprime_1'] += 1
    try:
        n_comp = len(entry.crystal.asymmetric_unit_molecule.components)
    except Exception:  # noqa: BLE001
        n_comp = -1
    if n_comp != 1:
        f['excluded_two_or_more_asu_components'] += 1
        return
    if d['z'] != d['n_ops']:
        f['excluded_z_ne_nops'] += 1
        return
    if d['sg_number'] is None or d['z_primitive'] is None:
        f['excluded_unknown_sg_or_centring'] += 1
        return
    f['organic_zp1'] += 1
    is_chnof = elements <= CHNOF
    if is_chnof:
        f['CHNOF_zp1'] += 1
    family = d['refcode'][:6]
    for name in POPULATIONS:
        if name == 'CHNOF_zp1' and not is_chnof:
            continue
        pop = state['pops'][name]
        first = family not in pop['families']
        pop['families'].add(family)
        sg = d['sg_number']
        add(pop, ('sg', sg), family, first)
        add(pop, ('z', d['z']), family, first)
        add(pop, ('zp', d['z_primitive']), family, first)
        if sg in DAY_26:
            add(pop, ('z26', d['z']), family, first)
            add(pop, ('zp26', d['z_primitive']), family, first)
            pop['settings'][(sg, d['sg_symbol'])] += 1


# -- checkpointing ---------------------------------------------------------

def _key_str(k):
    return f'{k[0]}|{k[1]}'


def _key_parse(s):
    a, b = s.split('|', 1)
    try:
        return (a, int(b)) if a in ('sg', 'z', 'zp', 'z26', 'zp26') else (int(a), b)
    except ValueError:
        return (a, b)


def save_state(state, path):
    out = {'n_scanned': state['n_scanned'], 'funnel': dict(state['funnel']), 'pops': {}}
    for name, pop in state['pops'].items():
        out['pops'][name] = {
            'raw': {_key_str(k): v for k, v in pop['raw'].items()},
            'dedup': {_key_str(k): v for k, v in pop['dedup'].items()},
            'families': sorted(pop['families']),
            'settings': {_key_str(k): v for k, v in pop['settings'].items()},
        }
    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(out, fh)
    os.replace(tmp, path)


def load_state(path):
    with open(path) as fh:
        raw = json.load(fh)
    state = new_state()
    state['n_scanned'] = raw['n_scanned']
    state['funnel'] = collections.Counter(raw['funnel'])
    for name, p in raw['pops'].items():
        pop = state['pops'][name]
        pop['raw'] = collections.Counter({_key_parse(k): v for k, v in p['raw'].items()})
        pop['dedup'] = collections.Counter({_key_parse(k): v for k, v in p['dedup'].items()})
        pop['families'] = set(p['families'])
        pop['settings'] = collections.Counter({_key_parse(k): v for k, v in p['settings'].items()})
    return state


# -- summarising -----------------------------------------------------------

def _dist(counter, tag):
    items = sorted((k[1], v) for k, v in counter.items() if k[0] == tag)
    total = sum(v for _, v in items)
    return ({str(k): v for k, v in items},
            {str(k): (v / total if total else 0.0) for k, v in items}, total)


def summarise_population(pop):
    out = {}
    for kind in ('raw', 'dedup'):
        c = pop[kind]
        sg_counts, sg_freq, n = _dist(c, 'sg')
        in26 = {str(g): c.get(('sg', g), 0) for g in DAY_26}
        n26 = sum(in26.values())
        z_counts, z_freq, _ = _dist(c, 'z')
        zp_counts, zp_freq, _ = _dist(c, 'zp')
        z26_counts, z26_freq, _ = _dist(c, 'z26')
        zp26_counts, zp26_freq, _ = _dist(c, 'zp26')
        out[kind] = {
            'n_structures': n,
            'space_group_counts': sg_counts,
            'space_group_freq': sg_freq,
            'day26': {
                'n_in_26': n26,
                'fraction_of_population_in_26': n26 / n if n else 0.0,
                'counts': in26,
                'freq_renormalised_over_26': {g: (v / n26 if n26 else 0.0) for g, v in in26.items()},
            },
            'z_conventional': {'counts': z_counts, 'freq': z_freq},
            'z_primitive': {'counts': zp_counts, 'freq': zp_freq},
            'z_conventional_within_26': {'counts': z26_counts, 'freq': z26_freq},
            'z_primitive_within_26': {'counts': zp26_counts, 'freq': zp26_freq},
        }
    out['n_raw'] = out['raw']['n_structures']
    out['n_families'] = out['dedup']['n_structures']
    settings = collections.defaultdict(dict)
    for (g, sym), v in sorted(pop['settings'].items()):
        settings[str(g)][sym] = v
    out['day26_setting_symbols_seen_raw'] = dict(settings)
    return out


def build_output(state, csd_meta, wall_s):
    import ccdc
    return {
        'metadata': {
            'description': "Bare CSD statistics for allocating Day-benchmark draws: "
                           "space group by IT number, Day-26 share, conventional and "
                           "primitive Z, over Z'=1 organic homomolecular crystals.",
            'script': 'energy_sampling/eval/day_bench/csd_statistics.py',
            'date': datetime.date.today().isoformat(),
            'ccdc_version': ccdc.__version__,
            **csd_meta,
            'filters': FILTERS,
            'not_filtered': ['molecular weight', 'rotatable bonds', 'R-factor',
                             'temperature', 'powder vs single-crystal'],
            'space_group_mapping': "IT number from Crystal.spacegroup_number_and_setting[0]; "
                                   "all settings of one group share its number (P2_1/c, "
                                   "P2_1/n, P2_1/a -> 14; C2/c, I2/a, A2/a -> 15; R-3 "
                                   "hexagonal or rhombohedral axes -> 148). Enantiomorphic "
                                   "pairs keep their separate numbers (76 vs 78, 96 vs 92, "
                                   "145 vs 144, 154 vs 152, 169 vs 170).",
            'z_primitive_definition': "conventional Z / number of lattice-centring "
                                      "translations (P 1; A,B,C,I 2; R obverse/reverse in "
                                      "hexagonal axes 3; F 4). At Z'=1 with Z = number of "
                                      "symmetry operators this equals the point-group order.",
            'dedup': "refcode family = first 6 characters of the refcode; the first "
                     "member passing the filters in database order is counted.",
            'day26_groups': list(DAY_26),
            'funnel_counts': dict(state['funnel']),
            'wall_time_s': round(wall_s, 1),
        },
        'populations': {name: summarise_population(state['pops'][name]) for name in POPULATIONS},
    }


def print_summary(out):
    for name in POPULATIONS:
        p = out['populations'][name]
        r, d = p['raw'], p['dedup']
        print(f"\nSpace-group frequency within Day's 26 groups, renormalised over the 26 "
              f"({name}; {p['n_raw']} structures raw, {p['n_families']} refcode families). "
              f"Fraction of population in the 26: raw {r['day26']['fraction_of_population_in_26']:.4f}, "
              f"families {d['day26']['fraction_of_population_in_26']:.4f}.")
        print(f"{'space group (IT no.)':>22} {'count raw':>10} {'freq raw':>9} "
              f"{'count fam':>10} {'freq fam':>9}")
        for g in DAY_26:
            g = str(g)
            print(f"{g:>22} {r['day26']['counts'][g]:>10} "
                  f"{r['day26']['freq_renormalised_over_26'][g]:>9.4f} "
                  f"{d['day26']['counts'][g]:>10} {d['day26']['freq_renormalised_over_26'][g]:>9.4f}")
        print(f"\nMolecules per primitive cell ({name}; all groups and within the 26; "
              f"fractions of structures).")
        keys = sorted(set(r['z_primitive']['counts']) | set(r['z_primitive_within_26']['counts']), key=int)
        print(f"{'Z primitive':>12} {'frac all raw':>13} {'frac all fam':>13} "
              f"{'frac 26 raw':>12} {'frac 26 fam':>12}")
        for k in keys:
            print(f"{k:>12} {r['z_primitive']['freq'].get(k, 0):>13.4f} "
                  f"{d['z_primitive']['freq'].get(k, 0):>13.4f} "
                  f"{r['z_primitive_within_26']['freq'].get(k, 0):>12.4f} "
                  f"{d['z_primitive_within_26']['freq'].get(k, 0):>12.4f}")


# -- known-refcode check ---------------------------------------------------

KNOWN = {
    # refcode: (IT number, conventional Z, Z', primitive Z)
    'IMAZOL03': (14, 4, 1.0, 4),   # P2_1/n setting of P2_1/c
    'URACIL': (14, 4, 1.0, 4),     # P2_1/a setting
    'ABACOW': (15, 8, 1.0, 4),     # C2/c
    'ABIDEY': (15, 8, 1.0, 4),     # I2/a setting of C2/c
    'AFAJID': (148, 18, 1.0, 6),   # R-3, hexagonal axes
    'ABADUE': (43, 16, 1.0, 4),    # Fdd2
    'BENZEN': (61, 4, 0.5, None),  # Pbca, molecule on inversion centre: Z'=1/2
}


def check(reader):
    ok = True
    for refcode, (sg, z, zp, zprim) in KNOWN.items():
        d = describe(reader.entry(refcode))
        got = (d['sg_number'], d['z'], d['z_prime'], d['z_primitive'] if d['z_prime'] == 1 else None)
        good = got == (sg, z, zp, zprim)
        ok &= good
        print(f"{refcode:>10} {d['sg_symbol']:>8} centring={d['lattice_centring']:<10} "
              f"(IT, Z, Z', Zprim) got={got} expected={(sg, z, zp, zprim)} {'OK' if good else 'MISMATCH'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  'csd_statistics.json'))
    ap.add_argument('--scratch', default=None, help='checkpoint directory')
    ap.add_argument('--checkpoint_every', type=int, default=50000)
    ap.add_argument('--limit', type=int, default=None, help='scan only the first N entries')
    ap.add_argument('--check', action='store_true', help='known-refcode check only')
    args = ap.parse_args()

    from ccdc import io
    reader = io.EntryReader('CSD')
    if not check(reader):
        raise SystemExit('known-refcode check failed')
    if args.check:
        return
    csd_meta = {'csd_n_entries': len(reader)}
    try:
        csd_meta['csd_data_version'] = io.csd_version()
        csd_meta['csd_directory'] = io.csd_directory()
    except Exception:  # noqa: BLE001
        pass

    ckpt = None
    if args.scratch:
        os.makedirs(args.scratch, exist_ok=True)
        ckpt = os.path.join(args.scratch, 'csd_statistics_ckpt.json')
    state = load_state(ckpt) if ckpt and os.path.exists(ckpt) else new_state()
    start = state['n_scanned']
    t0 = time.time()
    print(f'resuming at entry {start}' if start else 'starting', flush=True)
    for i, entry in enumerate(reader.entries()):
        if i < start:
            continue
        if args.limit is not None and i >= args.limit:
            break
        try:
            classify(entry, state)
        except Exception:  # noqa: BLE001 -- count, never lose the pass
            state['funnel']['entry_error'] += 1
        state['n_scanned'] = i + 1
        if ckpt and state['n_scanned'] % args.checkpoint_every == 0:
            save_state(state, ckpt)
            print(f"{state['n_scanned']} scanned, {time.time() - t0:.0f} s, "
                  f"organic_zp1 {state['funnel']['organic_zp1']}, "
                  f"CHNOF_zp1 {state['funnel']['CHNOF_zp1']}", flush=True)
    if ckpt:
        save_state(state, ckpt)
    out = build_output(state, csd_meta, time.time() - t0)
    with open(args.out, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'wrote {args.out}')
    print_summary(out)


if __name__ == '__main__':
    main()
