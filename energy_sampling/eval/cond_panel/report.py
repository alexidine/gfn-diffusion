"""
One page for a figure directory: every figure the suite wrote there, with its caption, the
findings read off it and the numbers it shows, grouped by question. Self-contained HTML (the
images are embedded), so the file can be opened or sent on its own.

`--notes` takes a text file of curated points to lead the page: lines starting with `# ` open
a list, `- ` lines are its items. They are shown as given; nothing in them is computed here.

    cd energy_sampling
    python -m eval.cond_panel.report --figures <dir>/figures --out <dir>/report.html [--notes notes.txt]
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import os

SECTIONS = (
    ('Which molecules the model handles well', ('quality_by_feature', 'group_effects', 'variance_explained')),
    ('Unseen molecules', ('neighbour_matched',)),
    ('What the crystals follow in the molecule', ('latent_feature_corr', 'model_vs_reference', 'latent_widths')),
    ('Between conditions', ('pairs_deep', 'pairs_which_latents', 'pairs_shallow')),
    ('Condition space and the level estimates', ('condition_map', 'levels')),
    ('Packing motifs', ('motif_prevalence', 'motif_agreement', 'motif_energy')),
    ('Other geometries of the same molecule', ('jitter_ladder', 'conformers', 'conformer_frames')),
    ('Draws relaxed to their minima', ('relax_outcomes', 'relax_spread', 'relax_two_relaxers')),
    ('Mock structure prediction', ('csp_recall',)),
)
CSS = """
:root { --surface: #fcfcfb; --page: #f9f9f7; --ink: #0b0b0b; --ink2: #52514e; --muted: #898781; --rule: #e1e0d9; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink); font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1180px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 24px; margin: 0 0 4px; }
h2 { font-size: 19px; margin: 40px 0 12px; padding-top: 16px; border-top: 1px solid var(--rule); }
h3 { font-size: 16px; margin: 0 0 8px; }
.meta { color: var(--ink2); margin: 0 0 20px; }
section.fig { background: var(--surface); border: 1px solid var(--rule); border-radius: 8px; padding: 16px; margin: 0 0 20px; }
section.fig img { max-width: 100%; height: auto; display: block; margin: 0 auto 10px; }
.caption { color: var(--ink2); font-size: 13.5px; margin: 0 0 10px; }
ul { margin: 0 0 10px; padding-left: 20px; }
li { margin: 2px 0; }
details { margin-top: 6px; }
summary { cursor: pointer; color: var(--ink2); font-size: 13.5px; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; font-size: 12.5px; margin-top: 8px; font-variant-numeric: tabular-nums; }
th, td { border-bottom: 1px solid var(--rule); padding: 3px 10px 3px 0; text-align: left; white-space: nowrap; }
th { color: var(--ink2); font-weight: 600; }
nav a { color: var(--ink2); margin-right: 14px; font-size: 13.5px; }
.notes { background: var(--surface); border: 1px solid var(--rule); border-radius: 8px; padding: 16px; }
"""


def notes_html(path):
    out, open_list = [], False
    for line in open(path, encoding='utf-8').read().splitlines():
        if line.startswith('# '):
            if open_list:
                out.append('</ul>')
            out.append(f'<h3>{html.escape(line[2:])}</h3><ul>')
            open_list = True
        elif line.startswith('- '):
            out.append(f'<li>{html.escape(line[2:])}</li>')
    if open_list:
        out.append('</ul>')
    return '<div class="notes">' + ''.join(out) + '</div>'


def figure_html(fig_dir, item):
    png = os.path.join(fig_dir, f'{item["name"]}.png')
    data = base64.b64encode(open(png, 'rb').read()).decode()
    parts = [f'<section class="fig" id="{item["name"]}"><h3>{html.escape(item["title"])}</h3>',
             f'<img alt="{html.escape(item["title"])}" src="data:image/png;base64,{data}">',
             f'<p class="caption">{html.escape(item["caption"])}</p>']
    if item.get('points'):
        parts.append('<ul>' + ''.join(f'<li>{html.escape(p)}</li>' for p in item['points']) + '</ul>')
    t = item.get('table')
    if t:
        head = ''.join(f'<th>{html.escape(str(h))}</th>' for h in t['header'])
        body = ''.join('<tr>' + ''.join(f'<td>{html.escape(str(v))}</td>' for v in row) + '</tr>' for row in t['rows'])
        parts.append(f'<details><summary>The numbers in this figure ({len(t["rows"])} rows)</summary><div class="scroll">'
                     f'<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div></details>')
    parts.append('</section>')
    return ''.join(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--figures', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--title', default='Per-condition evaluation')
    ap.add_argument('--subtitle', default='')
    ap.add_argument('--notes', default=None)
    args = ap.parse_args()
    items = {i['name']: i for i in json.load(open(os.path.join(args.figures, 'figures.json'), encoding='utf-8'))}
    body, nav, used = [], [], set()
    for k, (title, names) in enumerate(SECTIONS):
        present = [n for n in names if n in items]
        if not present:
            continue
        nav.append(f'<a href="#s{k}">{html.escape(title)}</a>')
        body.append(f'<h2 id="s{k}">{html.escape(title)}</h2>' + ''.join(figure_html(args.figures, items[n]) for n in present))
        used.update(present)
    rest = [n for n in items if n not in used]
    if rest:
        body.append('<h2>Other figures</h2>' + ''.join(figure_html(args.figures, items[n]) for n in rest))
    notes = notes_html(args.notes) if args.notes else ''
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>{html.escape(args.title)}</title><style>{CSS}</style></head><body><main><h1>{html.escape(args.title)}</h1>'
            f'<p class="meta">{html.escape(args.subtitle)}</p><nav>{"".join(nav)}</nav>{notes}{"".join(body)}</main></body></html>')
    with open(args.out, 'w', encoding='utf-8') as fh:
        fh.write(page)
    print(f'{args.out}: {len(used) + len(rest)} figures, {os.path.getsize(args.out) / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
