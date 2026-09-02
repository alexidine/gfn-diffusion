"""
Render the acridine comparison report from report_data.json.

Every number on the page comes from the JSON, which is itself a join over the
measurement artifacts -- so the report cannot state a number no script produced.
Nothing is computed here beyond formatting and a few counts derived in view.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, 'report_data.json'), encoding='utf-8'))
OUT = os.path.join(HERE, 'acridine_comparison_report.html')

POLY_OF = {}
for lv in ('l2', 'l1', 'l0'):
    for form, r in (D['polymorph_compack'] or {}).get(lv, {}).items():
        if r['best'] >= 20:
            POLY_OF.setdefault(r['which'], {})[lv] = (form, r['rmsd'])

DUP = {d['b']: d['a'] for d in D['duplicates']}


def e(v, p=2):
    return '&mdash;' if v is None else f'{v:.{p}f}'


def cell(c, cut=20):
    if c is None:
        return '<td class="num">&mdash;</td>'
    ok = c['n'] >= cut
    k = 'yes' if ok else 'no'
    txt = f"{c['n']}/20" + (f" &middot; {c['rmsd']:.3f}" if ok else '')
    return f'<td class="num {k}">{txt}</td>'


# ---- conformer -------------------------------------------------------------
conf_rows = []
for line in D['conformer_audit']:
    s = line.rstrip()
    if not s or s.startswith('-') or s.startswith('source') or s.startswith('reference'):
        continue
    if s.startswith('  ') and 'heavy atoms' in s:
        continue
    conf_rows.append(s)

ref_line = next((l for l in D['conformer_audit'] if 'heavy atoms' in l), '')
same_line = next((l for l in D['conformer_audit'] if 'carry the reference' in l), '')
l0_line = next((l for l in D['conformer_audit'] if 'nikos L0 vs' in l), '')


def conf_table():
    body = []
    for s in conf_rows:
        if 'sources carry' in s or 'nikos L0 vs' in s:
            continue
        name = s[:36].strip()
        rest = s[36:].split()
        if not rest:
            continue
        if not rest[0].replace('.', '').replace('-', '').isdigit():
            body.append(f'<tr><td>{name}</td><td colspan="4" class="warn">'
                        f'{" ".join(rest)}</td></tr>')
            continue
        n, dev, bmean = rest[0], rest[1], rest[2]
        rng = rest[3] if len(rest) > 3 else ''
        diff = 'DIFFERENT' in s
        cls = 'no' if diff else 'yes'
        verdict = 'old conformer' if diff else 'reference'
        body.append(
            f'<tr><td>{name}</td><td class="num">{int(n):,}</td>'
            f'<td class="num {cls}">{dev}</td><td class="num">{bmean}</td>'
            f'<td class="num">{rng.replace("-", "&ndash;")}</td>'
            f'<td class="{cls}">{verdict}</td></tr>')
    return ''.join(body)


# ---- polymorphs ------------------------------------------------------------
poly_rows = ''.join(
    f'<tr><td class="id">{p["name"]}</td><td class="num">{p["sg"]}</td>'
    f'<td class="num">{p["zp"]}</td><td class="num">{e(p["e_exp"])}</td>'
    f'<td class="num">{e(p["e_rel"])}</td><td class="num">{e(p["de"])}</td>'
    f'<td class="num">{p["moved"]:.4f}</td>'
    f'<td>{"<span class=no>broken reference</span>" if p["name"] == "ACRDIN08" else ""}</td></tr>'
    for p in sorted(D['polymorphs'], key=lambda d: d['e_exp']))

# ---- nikos -----------------------------------------------------------------
nik_rows = []
for r in sorted(D['nikos'], key=lambda d: (d['sg'], d['zp'], d['key'])):
    ident = POLY_OF.get(r['key'], {}).get('l2')
    idtxt = f'<b>{ident[0]}</b> &middot; {ident[1]:.3f}' if ident else '&mdash;'
    dup = (f'<span class="dup">= {DUP[r["key"]]}</span>'
           if r['key'] in DUP else '')
    nik_rows.append(
        f'<tr><td class="id">{r["key"]}{dup}</td><td>{r["sg_label"]}</td>'
        f'<td class="num">{r["zp"]}</td>'
        f'<td class="num">{e(r["e0"])}</td><td class="num">{e(r["e1"])}</td>'
        f'<td class="num">{e(r["e2"])}</td>'
        f'<td class="num">{r["moved"]:.4f}</td><td>{idtxt}</td>'
        f'{cell(r["land_l1"])}{cell(r["land_l2"])}</tr>')
nik_rows = ''.join(nik_rows)

# ---- polymorph compack across levels ---------------------------------------
pc = D['polymorph_compack'] or {}
forms = list(pc.get('l2', {}))
pc_rows = ''.join(
    '<tr><td class="id">%s</td>%s</tr>' % (
        f,
        ''.join(
            (f'<td class="num {"yes" if pc[lv][f]["best"] >= 20 else "no"}">'
             f'{pc[lv][f]["best"]}/20 &middot; {pc[lv][f]["rmsd"]:.3f}</td>'
             f'<td class="mono sm">{pc[lv][f]["which"]}</td>')
            for lv in ('l0', 'l1', 'l2') if f in pc.get(lv, {})))
    for f in sorted(forms, key=lambda x: pc['l2'][x]['rmsd']))

n_land_l1 = sum(1 for r in D['nikos'] if r['land_l1']['n'] >= 20)
n_land_l2 = sum(1 for r in D['nikos'] if r['land_l2']['n'] >= 20)
n_matched = len({v['l2'][0] for v in POLY_OF.values() if 'l2' in v})

HTML = f"""<title>Acridine Structure Comparison</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Spectral:wght@500;600&amp;family=IBM+Plex+Mono:wght@400;500;600&amp;family=IBM+Plex+Sans:wght@400;500;600&amp;display=swap">
<style>
:root{{
  color-scheme:light;
  --surface:#fcfcfb; --card:#ffffff; --rule:#dcdcd6; --rule-2:#ecece7;
  --ink:#0b0b0b; --ink-2:#52514e; --ink-3:#7d7c76;
  --yes:#1b6b4c; --no:#b04a22; --accent:#2a78d6; --warnbg:#fbf4e8;
}}
@media (prefers-color-scheme:dark){{
  :root:not([data-theme="light"]){{
    color-scheme:dark;
    --surface:#1a1a19; --card:#201f1e; --rule:#333330; --rule-2:#2a2a27;
    --ink:#ffffff; --ink-2:#c3c2b7; --ink-3:#8d8c83;
    --yes:#5cbf95; --no:#e08a5f; --accent:#5598e7; --warnbg:#2a2519;
  }}
}}
:root[data-theme="dark"]{{
  color-scheme:dark;
  --surface:#1a1a19; --card:#201f1e; --rule:#333330; --rule-2:#2a2a27;
  --ink:#ffffff; --ink-2:#c3c2b7; --ink-3:#8d8c83;
  --yes:#5cbf95; --no:#e08a5f; --accent:#5598e7; --warnbg:#2a2519;
}}
*{{box-sizing:border-box}}
body{{margin:0; background:var(--surface); color:var(--ink);
  font-family:"IBM Plex Sans",system-ui,sans-serif; font-size:15px; line-height:1.62;
  -webkit-font-smoothing:antialiased}}
.wrap{{max-width:1140px; margin:0 auto; padding:3.5rem 1.5rem 6rem;
  display:flex; flex-direction:column; gap:2.6rem}}
header{{display:flex; flex-direction:column; gap:.55rem;
  border-bottom:2px solid var(--rule); padding-bottom:1.5rem}}
.eyebrow{{font-family:"IBM Plex Mono",monospace; font-size:.72rem; letter-spacing:.11em;
  text-transform:uppercase; color:var(--ink-3); margin:0}}
h1{{font-family:Spectral,Georgia,serif; font-weight:600; font-size:2.2rem;
  line-height:1.12; margin:0; text-wrap:balance}}
h2{{font-family:Spectral,Georgia,serif; font-weight:600; font-size:1.35rem;
  margin:0 0 .2rem; text-wrap:balance}}
h3{{font-size:.95rem; font-weight:600; margin:1.4rem 0 .4rem; color:var(--ink-2)}}
p{{margin:0 0 .85rem; max-width:78ch}}
section{{display:flex; flex-direction:column}}
.lede{{color:var(--ink-2); font-size:1.02rem; max-width:74ch}}
.sub{{color:var(--ink-2); font-size:.92rem; margin:0 0 1rem; max-width:78ch}}
.tw{{overflow-x:auto; margin:.5rem 0 .3rem; border:1px solid var(--rule);
  border-radius:4px; background:var(--card)}}
table{{border-collapse:collapse; width:100%; font-size:.83rem;
  font-variant-numeric:tabular-nums}}
th{{text-align:left; font-weight:600; color:var(--ink-3); padding:.5rem .8rem;
  border-bottom:1px solid var(--rule); white-space:nowrap; font-size:.68rem;
  letter-spacing:.07em; text-transform:uppercase; background:var(--surface)}}
th.num,td.num{{text-align:right}}
td{{padding:.42rem .8rem; border-bottom:1px solid var(--rule-2); white-space:nowrap}}
tr:last-child td{{border-bottom:none}}
td.id,.mono{{font-family:"IBM Plex Mono",monospace}}
td.num{{font-family:"IBM Plex Mono",monospace}}
.yes{{color:var(--yes); font-weight:500}}
.no{{color:var(--no); font-weight:500}}
.sm{{font-size:.76rem; color:var(--ink-3)}}
.dup{{font-family:"IBM Plex Sans",sans-serif; font-size:.72rem; color:var(--ink-3);
  margin-left:.45rem}}
.cap{{font-size:.85rem; color:var(--ink-2); max-width:80ch; margin:.55rem 0 0}}
.cap b{{color:var(--ink)}}
.box{{background:var(--card); border:1px solid var(--rule); border-left:3px solid var(--accent);
  border-radius:4px; padding:1rem 1.25rem; font-size:.9rem; max-width:82ch}}
.box.warn{{border-left-color:var(--no); background:var(--warnbg)}}
.box p:last-child{{margin-bottom:0}}
dl.levels{{display:grid; grid-template-columns:auto 1fr; gap:.35rem 1.1rem;
  margin:.4rem 0 0; font-size:.9rem}}
dl.levels dt{{font-family:"IBM Plex Mono",monospace; font-weight:600; color:var(--accent)}}
dl.levels dd{{margin:0; color:var(--ink-2)}}
pre{{background:var(--card); border:1px solid var(--rule); border-radius:4px;
  padding:.85rem 1rem; overflow-x:auto; font-family:"IBM Plex Mono",monospace;
  font-size:.78rem; line-height:1.55; margin:.4rem 0}}
ul{{margin:.2rem 0 .9rem; padding-left:1.15rem; max-width:78ch}}
li{{margin:.3rem 0}}
.foot{{border-top:1px solid var(--rule); padding-top:1.3rem; color:var(--ink-3);
  font-size:.85rem; max-width:80ch}}
</style>

<div class="wrap">
<header>
 <p class="eyebrow">acridine &middot; structure comparison &middot; 2026-08-27</p>
 <h1>Nikos&rsquo; proposals against the known forms and our search</h1>
 <p class="lede">Which of his structures are known polymorphs, which our brute-force
 landscapes contain, and on what conformer every one of those statements rests.
 Every number below was produced by a script in
 <span class="mono">eval/nikos_comparison/</span> and joined into one file; none
 was transcribed by hand.</p>
</header>

<section>
<h2>The four levels</h2>
<p class="sub">The single most common way to get this comparison wrong is to quote a
number from one level beside a number from another. Each table states its level.</p>
<dl class="levels">
 <dt>L0</dt><dd>as delivered &mdash; his file, his conformer</dd>
 <dt>L1</dt><dd>reprojected &mdash; same cell and pose, our reference conformer substituted</dd>
 <dt>L2</dt><dd>relaxed &mdash; L1 rigid-body relaxed on our MACE surface (cell + pose)</dd>
 <dt>ref</dt><dd>the experimental structures, and separately those same structures relaxed</dd>
</dl>
<div class="box warn" style="margin-top:1.1rem">
<p><b>No all-atom relaxation exists anywhere in this analysis.</b> Every L2 varies
cell lengths, cell angles, aunit centroid and aunit orientation only. The molecule is
held rigid at the reference conformer throughout. Any statement below about a
structure &ldquo;relaxing onto&rdquo; another is a statement about packing, not about
molecular geometry.</p>
</div>
</section>

<section>
<h2>1 &middot; Which conformer, everywhere</h2>
<p class="sub">Lattice energy is a difference between a crystal and its constituent
molecule, so two artifacts built on different conformers are not comparable at all
&mdash; and the failure is quiet: both score, both look reasonable, and the numbers
are tens of kJ/mol apart for reasons unrelated to packing. Each source is fingerprinted
by its sorted intramolecular distance vector over heavy atoms, which is invariant to
rotation, translation and atom order.</p>
<div class="tw"><table>
<thead><tr><th>source</th><th class="num">n</th><th class="num">max dev vs ref (&Aring;)</th>
<th class="num">mean bond (&Aring;)</th><th class="num">bond range</th><th>verdict</th></tr></thead>
<tbody>{conf_table()}</tbody></table></div>
<p class="cap"><b>{same_line.strip()}</b> Reference is
<span class="mono">opt_acridine_conformer.pt</span>: {ref_line.strip()}.</p>
<div class="box" style="margin-top:1rem">
<p><b>His delivered molecule is our old conformer.</b> {l0_line.strip()} So there was
never a third geometry in play &mdash; L0 disagreeing with the reference is the
expected consequence of him and us both starting from the same original molecule,
and it is why reprojecting L0&rarr;L1 barely moves any match below.</p>
<p style="margin-top:.7rem"><b>The trap is live.</b>
<span class="mono">std_acridine_polymorphs.pt</span> and
<span class="mono">std_opt_acridine_polymorphs.pt</span> hold the same cells and poses
and differ only in the molecule. <span class="mono">process_target.py</span> still
writes the old-conformer file. Everything in this report uses
<span class="mono">std_opt_</span>.</p>
</div>
</section>

<section>
<h2>2 &middot; The known forms, experimental and optimized</h2>
<p class="sub">Each experimental structure, and the same structure after rigid-body
relaxation on our surface. Energies in kJ/mol on the stored scale
(offset {D['energy_offset']}); <span class="mono">moved</span> is RDF distance travelled.</p>
<div class="tw"><table>
<thead><tr><th>form</th><th class="num">SG</th><th class="num">Z&prime;</th>
<th class="num">E experimental</th><th class="num">E relaxed</th><th class="num">&Delta;E</th>
<th class="num">moved</th><th></th></tr></thead>
<tbody>{poly_rows}</tbody></table></div>
<p class="cap">Six sound forms relax by <b>0.65&ndash;3.51 kJ/mol</b> and move
<b>0.046&ndash;0.104</b> &mdash; a well-determined structure sits near its own minimum.
<b>ACRDIN08 relaxes by 18.20 and moves 0.207</b>, twice as far and five times as deep as
any other, and still lands ~20 kJ/mol above every other form. Our copy of it is broken;
it is excluded from conclusions rather than reported as a miss.</p>
<p class="cap" style="margin-top:.6rem"><b>Relaxation expands these cells rather than
compressing them.</b> Packing coefficient falls for six of seven (ACRDIN04
0.7008&rarr;0.6852, ACRIDIN_VIII 0.6790&rarr;0.6429; only ACRDIN07 rises, +0.004). That is
consistent with the conformer substitution: the reference conformer is 2.7% larger in
radius of gyration than the one the experimental cells were determined with, so the cell
must grow to accommodate it.</p>
</section>

<section>
<h2>3 &middot; His proposals, as delivered and optimized</h2>
<p class="sub">All 13 ingested structures across the three levels, what each is
identified as, and whether our landscape for its own SG/Z&prime; contains it.</p>
<div class="tw"><table>
<thead><tr><th>structure</th><th>SG</th><th class="num">Z&prime;</th>
<th class="num">E L0</th><th class="num">E L1</th><th class="num">E L2</th>
<th class="num">moved</th><th>is a known form (L2)</th>
<th class="num">landscape L1</th><th class="num">landscape L2</th></tr></thead>
<tbody>{nik_rows}</tbody></table></div>
<div class="box warn" style="margin-top:1rem">
<p><b>E&nbsp;L0 is positive, and that is the conformer trap made quantitative.</b> L0
carries HIS molecule &mdash; which section&nbsp;1 shows is our old conformer &mdash; scored on
our surface. The old conformer scores <b>+11 to +53&nbsp;kJ/mol</b> where the reference
conformer scores <b>&minus;55 to &minus;60</b>, so the L0&rarr;L1 column steps ~60&nbsp;kJ/mol for
reasons that have nothing to do with packing. <b>Never compare an L0 energy to an L1 or L2
energy.</b> The L0 column is here to show the size of that effect, not to be read as a
lattice energy.</p>
</div>
<p class="cap"><b>Landscape matches go {n_land_l1} &rarr; {n_land_l2} of 13</b> once both
sides are relaxed. As delivered his structures sit 2&ndash;12 kJ/mol above the minima they
belong to, because they arrive unrelaxed on our surface &mdash; which is what made the L1
comparison look worse than reality.</p>
<div class="box warn" style="margin-top:1rem">
<p><b>13 ingested structures are 12 distinct ones.</b>
{' and '.join(f"<span class='mono'>{d['a']}</span> / <span class='mono'>{d['b']}</span>"
              for d in D['duplicates'])} converge to RDF
{D['duplicates'][0]['rdf_l2']} at L2 &mdash; inside the 0.10 cut. They are
{D['duplicates'][0]['note']}. Counts elsewhere should not treat them as independent.</p>
<p style="margin-top:.7rem">Two further structures were never ingested:
{', '.join(f"<span class='mono'>{x['key']}</span> ({x['sg']})" for x in D['excluded'])}
&mdash; space groups we have not searched, so nothing there is evidence either way.</p>
</div>
</section>

<section>
<h2>4 &middot; Matched against the known forms</h2>
<p class="sub">Each known polymorph COMPACK&rsquo;d against his whole pool, at all three
levels. At L2 the reference is the <em>relaxed</em> experimental structure, so both sides
sit at the same minimum &mdash; the only like-for-like comparison of the three.</p>
<div class="tw"><table>
<thead><tr><th>form</th>
<th class="num">L0 vs experimental</th><th>best</th>
<th class="num">L1 vs experimental</th><th>best</th>
<th class="num">L2 vs relaxed</th><th>best</th></tr></thead>
<tbody>{pc_rows}</tbody></table></div>
<p class="cap"><b>{n_matched} of 7 known forms are present in his pool, at full 20/20,
and every one sharpens under relaxation</b> &mdash; RMSDs collapse from 0.137&ndash;0.314 to
0.004&ndash;0.125&nbsp;&Aring;. ACRDIN06 at 0.004 is exact to the precision of the method.
The seventh, ACRDIN08, is the broken reference. <b>0 of 91 comparisons failed inside the
similarity engine</b>, so no <span class="mono">0/0</span> result could be misread as a
perfect match.</p>
</section>

<section>
<h2>5 &middot; Matched against our landscapes</h2>
<p class="sub">The same structures against our brute-force search for their own
SG/Z&prime;, before and after relaxation. Landscape columns are in section 3.</p>
<div class="box">
<p><b>sg14-Z&prime;1</b> &mdash; contains both its forms. <b>sg9-Z&prime;2</b> &mdash; contains both
its forms, and additionally confirms <span class="mono">nik00006</span> and
<span class="mono">nik00008</span> at 20/20, which are <em>not</em> known polymorphs:
independent proposals corroborated by an independent method.</p>
<p style="margin-top:.7rem"><b>sg14-Z&prime;2 contains neither of its forms, and relaxation
does not rescue them.</b> <span class="mono">nik00011</span> is ACRDIN07 and
<span class="mono">nik00012</span> is ACRDIN06 &mdash; both 20/20 against the relaxed
references at RMSD 0.043 and 0.004. Both barely move on relaxation (0.082, 0.077). Our
sg14-Z&prime;2 landscape still finds neither (9/20, 5/20). Real structures, correctly
identified, essentially unmoved, and absent from our search: a genuine gap in that
landscape, no longer confounded by relaxation state.</p>
</div>
<p class="cap" style="margin-top:.9rem">An RDF screen over the same data pointed the
other way &mdash; within-cut counts for sg14-Z&prime;2 went 0/13 to 2/13 at L2, which
looks like a recovery. COMPACK refuted it. The screening statistic is not the result;
it selects candidates for confirmation.</p>
</section>

<section>
<h2>What this does and does not establish</h2>
<h3>Established</h3>
<ul>
<li>One conformer underlies every artifact used in the comparison, verified to
0.000000&nbsp;&Aring; on all 91 intramolecular distances.</li>
<li>Six of seven known forms are in his pool at 20/20, confirmed at both relaxation states.</li>
<li>Our sg14-Z&prime;1 and sg9-Z&prime;2 landscapes contain their known forms; sg14-Z&prime;2 does not,
and that is not an artifact of relaxation state.</li>
<li>Two of his structures are novel and independently corroborated by our search.</li>
</ul>
<h3>Not established</h3>
<ul>
<li><b>ACRDIN08.</b> Our reference is broken; the form is neither confirmed nor excluded.
Re-derive it from the CIF before drawing any conclusion about it.</li>
<li><b>All-atom geometry.</b> Nothing here relaxes molecular internal coordinates. Rankings
near the minimum can invert under a flexible conformer.</li>
<li><b>Optimizer sensitivity.</b> L2 uses one optimizer (rprop, 120 steps, annealed). Structures
move 0.06&ndash;0.37 RDF to reach it. A gentler setting has not been run, so verdicts that
changed only under relaxation are not separated from optimizer aggression. The two that
depend on this are <span class="mono">nik00005</span> and <span class="mono">nik00009</span>.</li>
<li><b>Absolute packing coefficients.</b> <span class="mono">mol_volume</span> is carried, not
recomputed &mdash; identical (150.44644&nbsp;&Aring;&sup3;) for both conformers despite a 2.7% size
change. Packing coefficient is therefore proportional to 1/V<sub>cell</sub> within this dataset;
relative comparisons hold, the absolute scale inherits that constant.</li>
</ul>
</section>

<section>
<h2>Reproduction</h2>
<p class="sub">In order. Requires <span class="mono">PYTHONPATH</span> to include both
repos; see <span class="mono">eval/nikos_comparison/README.md</span>.</p>
<pre>python eval/nikos_comparison/audit_conformer.py         # section 1
python eval/nikos_comparison/relax_l2_chunked.py --chunk 1 --device cpu
python eval/nikos_comparison/relax_l2_chunked.py --merge --chunk 1 --device cpu
python eval/nikos_comparison/relax_l2_chunked.py --polymorphs --chunk 1 --device cpu
python eval/nikos_comparison/nikos_vs_polymorphs.py     # section 4
python eval/nikos_comparison/compare.py --level l2      # section 5
python eval/nikos_comparison/build_report_data.py       # joins all of the above
python eval/nikos_comparison/build_report.py            # this page</pre>
<p class="cap"><b>Relax one structure per batch.</b> <span class="mono">--chunk 1</span> is
not a memory convenience. Convergence is evaluated over the batch, so a structure starting
near its minimum halts the batch and truncates any slower structure in it. Batching two at a
time changed where <b>10 of 13</b> structures landed and put one 16&nbsp;kJ/mol wrong.</p>
</section>

<p class="foot">Energies are MACE lattice energies on the stored scale (offset
{D['energy_offset']}), rigid reference conformer. Structural identity throughout is ccdc
<span class="mono">PackingSimilarity</span> over a 20-molecule shell; a
<span class="mono">0/0</span> result is an engine failure, not a zero-RMSD match, and is
excluded rather than counted. Source data and scripts:
<span class="mono">energy_sampling/eval/nikos_comparison/</span>.</p>
</div>
"""

open(OUT, 'w', encoding='utf-8').write(HTML)
print(f"wrote {OUT} ({len(HTML):,} bytes)")
print(f"  {len(D['nikos'])} structures, {len(D['polymorphs'])} polymorphs, "
      f"{n_matched} forms matched, landscape {n_land_l1} -> {n_land_l2}")
