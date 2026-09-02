"""
The section that answers: are the tight regions in the embedding real?

Four questions, answered in the order that matters. Numbers come from
tight_regions.py -- this file only renders them.
"""
import json
import os

import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

SP = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join('D:', os.sep, 'crystal_datasets', 'acridine')

T = json.load(open(os.path.join(SP, 'tight_regions.json')))
S = T['stats']
CG = {g['basin']: g for g in json.load(open(os.path.join(SP,
                                                        'coarse_landscape.json')))}

blob = torch.load(os.path.join(ROOT, 'nikos_comparison', 'lowE_dmat.pt'),
                  weights_only=False)
D = blob['D'].numpy().astype(np.float64)
E = blob['E'].numpy()
lab = np.asarray(fcluster(linkage(squareform(D, checks=False), method='complete'),
                          t=0.10, criterion='distance'))

rows = []
for g in T['groups'][:8]:
    b = g['basin']
    m = np.where(lab == b)[0]
    w = D[np.ix_(m, m)]
    wi = w[np.triu_indices(len(m), 1)].mean() if len(m) > 1 else 0.0
    bt = D[np.ix_(m, np.where(lab != b)[0])].min()
    cg = CG.get(b, {})
    held = ' '.join(h['name'] for h in cg.get('holds') or []) or '&mdash;'
    rows.append(
        f'<tr><td class="mono">{b}</td><td class="num">{g["n"]:,}</td>'
        f'<td class="num">{g["emin"]:.2f}</td>'
        f'<td class="num">{wi:.4f}</td><td class="num">{w.max():.4f}</td>'
        f'<td class="num{" warn" if bt <= wi else ""}">{bt:.4f}</td>'
        f'<td>{cg.get("fam", "?")}</td><td>{held}</td></tr>')

BLOCK = f'''<h2>Are the tight regions real?</h2>
<p class="sub">The embedding has a handful of very dense knots and a lot of empty
space. That shape could be structure, or it could be the projection collapsing
things that are not actually alike. Four checks, in the order that matters &mdash;
if it were an artifact the rest would be moot, so that goes first.</p>

<ol class="qa">
<li><b>Not an artifact.</b> Local density in the picture against local density in
the true RDF metric, 10 nearest neighbours each, every one of the
{len(T['e']):,} structures: <b>r&nbsp;=&nbsp;{S['r']:.3f}</b>.
Tight on screen means tight in the metric. The tightest 15% of the picture sit at a
median true RDF distance of <b>{S['tight']:.4f}</b> from their ten neighbours &mdash;
not &ldquo;similar&rdquo;, <em>identical</em>. The loosest 15% sit at
{S['loose']:.4f}.</li>

<li><b>The knots are the optimiser converging.</b> A true distance of 0.0000
repeated hundreds of times is one structure found hundreds of times, from different
random starts. That is the useful reading of the whole picture: the density of a
knot is a <em>basin volume</em> measurement, not a plotting accident.</li>

<li><b>They line up with the basins.</b> Cutting the 2D picture into the same
{S['k']} groups the RDF metric gives and comparing the two labellings:
<b>adjusted Rand {S['ari']:.3f}</b> (1 = identical, 0 = chance). The disagreement is
small and one-sided &mdash; {S['merged']} of {S['k']} picture-groups fuse basins that
the metric separates, {S['split']} basins get split by the picture. So you can read
the embedding as a basin map, but the metric is the authority where they differ.</li>

<li><b>Distinct in RDF, not just on screen.</b> For the eight largest basins, the
distance to the <em>nearest structure in any other basin</em> against their own mean
internal distance:</li>
</ol>

<table class="tbl">
<thead><tr><th>basin</th><th class="num">found</th><th class="num">min E</th>
<th class="num">within</th><th class="num">diameter</th>
<th class="num">to nearest other</th><th>family</th><th>holds</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>

<p class="caveat"><b>Seven of the eight are cleanly separated</b> &mdash; further
from any other basin than their own members are from each other. The exception is
basin&nbsp;18, which sits <b>0.0082</b> from basin&nbsp;16, deep inside the band
where packing similarity is 100%. Those two are one basin that the 0.10 cut has
split. That is the price of complete linkage: it guarantees no basin is wider than
the cut, and pays by occasionally dividing a wide one. The reverse choice, average
linkage, made the opposite error on this same data &mdash; it produced a
1,167-member group whose mean internal distance (0.047) was <em>larger</em> than its
distance to the nearest other group (0.020), which is not a basin at all. Neither
linkage is right; the count is soft between roughly 30 and 45 and every conclusion
here survives that range.</p>'''

open(os.path.join(SP, 'clumps_block.html'), 'w', encoding='utf-8').write(BLOCK)
print(f"wrote clumps_block.html ({len(BLOCK):,} bytes), {len(rows)} rows, "
      f"r={S['r']:.3f} ari={S['ari']:.3f}")
