"""Render a workflow diagram from its YAML source into a self-contained HTML viewer.

    python docs/workvis/render.py unconditional          # unconditional.yaml -> unconditional.html
    python docs/workvis/render.py unconditional --baseline prev/unconditional.yaml

The YAML is the source an agent edits; the HTML is generated and never edited by
hand. Text between single dollars is TeX: the viewer typesets it (KaTeX to MathML)
BEFORE Mermaid lays the graph out, so every box is sized to its typeset math.
Mermaid's own $$ math cannot be used: its pattern is greedy, so a box holding two
formulas fails to parse.

A VARIANT (`extends:`) is drawn merged onto its base (diagram.py).

HIGHLIGHTS ARE A DIFF AGAINST THE BASELINE: for a base diagram, by default
prev/<workflow>.yaml, the copy /workvis takes before it revises the source; for a
variant, by default its merged base, so its highlights show what differs between the
two routes. A node that is new or differs there is outlined; within it, each line (and the title) not present in the
baseline node is bold and coloured. New edges are coloured. Removed nodes are
listed in the viewer's panel. No baseline file = nothing highlighted.

The page checks itself once drawn: `window.WORKVIS_CHECK` = {ok, nodes, expected,
mathErrors, error}, so a refresh confirms the render with one script call, no screenshot.

Resolution is yaml and string work only: no torch, no import of the trainer.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import diagram  # noqa: E402
MATH = re.compile(r'(\$[^$]+\$)')

#: node kind -> (Mermaid open, close, fill, stroke, text colour, dashed)
KINDS = {
    'step':     ('["', '"]',   '#ffffff', '#94a3b8', '#0f172a', False),
    'once':     ('["', '"]',   '#eef2ff', '#818cf8', '#0f172a', False),
    'gate':     ('["', '"]',   '#fffbeb', '#f59e0b', '#0f172a', False),
    'periodic': ('["', '"]',   '#f0fdfa', '#14b8a6', '#0f172a', False),
    'store':    ('[("', '")]', '#f0f9ff', '#0284c7', '#0f172a', False),
    'param':    ('{{"', '"}}', '#f5f3ff', '#7c3aed', '#0f172a', False),
    'inert':    ('["', '"]',   '#f8fafc', '#94a3b8', '#64748b', True),
    'warn':     ('["', '"]',   '#fef2f2', '#dc2626', '#0f172a', False),
    'note':     ('["', '"]',   '#ffffff', '#cbd5e1', '#334155', True),
}
#: group style -> (fill, stroke)
GROUP_STYLES = {
    'init': ('#f8fafc', '#cbd5e1'),
    'mle':  ('#f0fdf4', '#86efac'),
    'eq':   ('#fefce8', '#fde047'),
    'step': ('#fffbeb', '#fcd34d'),
    'plain': ('#ffffff', '#e2e8f0'),
}
#: edge kind -> Mermaid arrow
ARROWS = {'flow': '-->', 'data': '-.->', 'write': '==>'}
CHANGED = '#ea580c'


def load(path: Path) -> dict:
    with open(path, encoding='utf-8') as fh:
        return yaml.safe_load(fh)


def esc(text: str) -> str:
    """Plain text for a quoted Mermaid label. Math segments pass through as TeX."""
    out = []
    for part in MATH.split(text):
        if len(part) > 1 and part.startswith('$') and part.endswith('$'):
            if '"' in part:
                raise ValueError(f'a double quote inside math breaks the label: {part!r}')
            out.append(part)
        else:
            out.append(part.replace('"', '#quot;').replace('<', '#lt;').replace('>', '#gt;'))
    return ''.join(out)


def mark(html_text: str, changed: bool) -> str:
    return f"<span class='chg'>{html_text}</span>" if changed else html_text


def node_label(node: dict, base: dict | None, diffing: bool) -> tuple[str, bool]:
    """The HTML label, and whether the node counts as changed against its baseline."""
    lines = node.get('lines') or []
    if not diffing:
        changed, title_changed, old_lines = False, False, None
    elif base is None:
        changed, title_changed, old_lines = True, True, set()
    else:
        old_lines = set(base.get('lines') or [])
        title_changed = base.get('title') != node.get('title')
        changed = (title_changed or base.get('kind') != node.get('kind')
                   or (base.get('lines') or []) != lines)
    rows = [mark(f"<span class='t'>{esc(node['title'])}</span>", title_changed)]
    for line in lines:
        rows.append(mark(f"<span class='l'>{esc(line)}</span>",
                         old_lines is not None and line not in old_lines))
    return '<br/>'.join(rows), changed


def edge_key(e: dict) -> tuple:
    return (e['from'], e['to'], e.get('label', ''), e.get('kind', 'flow'))


def build(src: dict, base: dict | None) -> tuple[str, list[str], list[dict]]:
    """Mermaid source, ids of changed nodes, removed nodes."""
    diffing = base is not None
    base_nodes = {n['id']: n for n in (base or {}).get('nodes', [])}
    base_edges = {edge_key(e) for e in (base or {}).get('edges', [])}
    nodes = src['nodes']
    ids = {n['id'] for n in nodes}

    out = ['flowchart TD']
    changed_ids, kind_of = [], {}
    by_group: dict[str | None, list[str]] = {}
    for n in nodes:
        kind = n.get('kind', 'step')
        if kind not in KINDS:
            raise ValueError(f"node {n['id']}: unknown kind {kind!r}")
        opn, cls, *_ = KINDS[kind]
        label, changed = node_label(n, base_nodes.get(n['id']), diffing)
        if changed:
            changed_ids.append(n['id'])
        kind_of[n['id']] = kind
        by_group.setdefault(n.get('group'), []).append(f"  n_{n['id']}{opn}{label}{cls}")

    groups = src.get('groups', [])
    children: dict[str | None, list[dict]] = {}
    for g in groups:
        children.setdefault(g.get('parent'), []).append(g)

    def emit(parent: str | None, depth: int):
        for g in children.get(parent, []):
            out.append(f"{'  ' * depth}subgraph g_{g['id']}[\"<span class='gt'>{esc(g['title'])}</span>\"]")
            out.extend('  ' * depth + line for line in by_group.get(g['id'], []))
            emit(g['id'], depth + 1)
            out.append(f"{'  ' * depth}end")

    out.extend(by_group.get(None, []))
    emit(None, 1)

    new_edges = []
    for i, e in enumerate(src.get('edges', [])):
        for end in ('from', 'to'):
            if e[end] not in ids:
                raise ValueError(f"edge {e['from']} -> {e['to']}: no node {e[end]!r}")
        arrow = ARROWS[e.get('kind', 'flow')]
        lab = f'|"{esc(e["label"])}"|' if e.get('label') else ''
        out.append(f"  n_{e['from']} {arrow}{lab} n_{e['to']}")
        if diffing and edge_key(e) not in base_edges:
            new_edges.append(i)

    for kind, (_, _, fill, stroke, color, dashed) in KINDS.items():
        dash = ',stroke-dasharray:5 4' if dashed else ''
        out.append(f'  classDef {kind} fill:{fill},stroke:{stroke},color:{color},stroke-width:1.5px{dash}')
    out.append(f'  classDef changed stroke:{CHANGED},stroke-width:3.5px')
    for kind in KINDS:
        members = [f'n_{i}' for i, k in kind_of.items() if k == kind]
        if members:
            out.append(f"  class {','.join(members)} {kind}")
    if changed_ids:
        out.append(f"  class {','.join('n_' + i for i in changed_ids)} changed")
    for g in groups:
        fill, stroke = GROUP_STYLES[g.get('style', 'plain')]
        out.append(f'  style g_{g["id"]} fill:{fill},stroke:{stroke},stroke-width:1.5px')
    for i in new_edges:
        out.append(f'  linkStyle {i} stroke:{CHANGED},stroke-width:3px')

    removed = [n for i, n in base_nodes.items() if i not in ids]
    return '\n'.join(out), changed_ids, removed


def page(src: dict, mermaid_src: str, changed: list[str], removed: list[dict],
         base_label: str | None) -> str:
    titles = {n['id']: n['title'] for n in src['nodes']}
    meta = src.get('stamp', {})
    panel = {
        'title': src.get('title', src.get('workflow', '')),
        'subtitle': src.get('subtitle', ''),
        'commit': meta.get('commit', '?'),
        'audited': meta.get('audited', 'not audited'),
        'rendered': dt.datetime.now().strftime('%Y-%m-%d %H:%M'),
        'baseline': base_label,
        'overrides': [f'{k} = {v}' for k, v in (src.get('route', {}).get('overrides') or {}).items()],
        'changed': [{'id': i, 'title': titles[i]} for i in changed],
        'removed': [n['title'] for n in removed],
        'legend': [[k, KINDS[k][2], KINDS[k][3], KINDS[k][5]] for k in KINDS],
    }
    data = json.dumps({'mermaid': mermaid_src, 'panel': panel, 'n_nodes': len(src['nodes'])}).replace('</', '<\\/')
    return TEMPLATE.replace('__DATA__', data).replace('__TITLE__', panel['title'])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('workflow', help='name of docs/workvis/<workflow>.yaml')
    ap.add_argument('--baseline', help='YAML to diff against (default: prev/<workflow>.yaml; a variant, its base)')
    ap.add_argument('--out', help='output HTML (default docs/workvis/<workflow>.html)')
    args = ap.parse_args(argv)

    src = diagram.load(args.workflow)
    if args.baseline:
        base = load(Path(args.baseline))
        if base.get('extends'):
            base = diagram.merge(diagram.load(base['extends']), base)
    elif src.get('extends'):
        base = diagram.load(src['extends'])
    else:
        base_path = HERE / 'prev' / f'{args.workflow}.yaml'
        base = load(base_path) if base_path.exists() else None
    mermaid_src, changed, removed = build(src, base)
    base_label = None
    if base is not None:
        b = base.get('stamp', {})
        base_label = f"Changed since {b.get('commit', '?')} ({b.get('date', '?')})"
        if base.get('workflow') != src.get('workflow'):  # a sibling route's diagram: highlights = what differs
            base_label = f"Differs from {base.get('title', base.get('workflow'))} ({b.get('commit', '?')})"
    out = Path(args.out) if args.out else HERE / f'{args.workflow}.html'
    out.write_text(page(src, mermaid_src, changed, removed, base_label), encoding='utf-8')
    print(f'{out}: {len(src["nodes"])} nodes, {len(changed)} changed, {len(removed)} removed'
          + ('' if base is not None else ' (no baseline: nothing highlighted)'))
    return 0


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.min.js"></script>
<style>
  :root { --ink:#0f172a; --muted:#64748b; --line:#e2e8f0; --chg:#ea580c; --bg:#f1f5f9; }
  html, body { margin:0; height:100%; background:var(--bg); font-family:Inter, system-ui, sans-serif; color:var(--ink); }
  #stage { position:fixed; inset:0; overflow:hidden; cursor:grab; }
  #stage.dragging { cursor:grabbing; }
  #stage svg { width:100%; height:100%; display:block; }
  .node .label { font-family:Inter, system-ui, sans-serif; line-height:1.5; }
  .t { font-weight:650; font-size:15.5px; color:var(--ink); }
  .l { font-size:14px; color:#334155; }
  .gt { font-weight:700; font-size:17px; letter-spacing:.01em; color:#334155; }
  .chg, .chg .t, .chg .l, .chg math { color:var(--chg) !important; font-weight:700; }
  math { font-size:1.06em; }
  .node rect { rx:10px; ry:10px; }
  .cluster rect { rx:16px; ry:16px; }
  .edgeLabel { background:#fff !important; color:var(--muted); font-size:13px; padding:1px 4px; border-radius:4px; }
  .edgeLabel p, .edgeLabel span { background:#fff !important; }
  /* Overview (zoomed out): titles only, enlarged; strokes keep their screen width. */
  .lod-titles, .lod-marks { display:none; }
  svg.lod-over .lod-titles, svg.lod-over .lod-marks { display:inline; }
  svg.lod-over .node .label, svg.lod-over .edgeLabels, svg.lod-over .cluster-label { visibility:hidden; }
  svg.lod-over .label-container, svg.lod-over .cluster rect, svg.lod-over .flowchart-link { vector-effect:non-scaling-stroke; }
  .lodt { width:100%; height:100%; display:flex; align-items:center; justify-content:center; box-sizing:border-box; padding:0 8%; cursor:zoom-in; }
  .lodt > div { min-width:0; max-width:100%; text-align:center; font-weight:650; line-height:1.16; color:var(--ink); overflow-wrap:break-word; font-family:Inter, system-ui, sans-serif; }
  .lodt.changed > div { color:var(--chg); font-weight:700; }
  .lod-marks text { font-family:Inter, system-ui, sans-serif; font-weight:750; fill:#334155; opacity:.42; letter-spacing:.01em; }
  #panel { position:fixed; top:12px; left:12px; width:300px; max-height:calc(100% - 24px); overflow:auto;
           background:rgba(255,255,255,.96); border:1px solid var(--line); border-radius:12px;
           box-shadow:0 4px 18px rgba(15,23,42,.08); padding:14px 16px; font-size:13px; }
  #panel.collapsed > :not(header) { display:none; }
  #panel header { display:flex; justify-content:space-between; align-items:baseline; gap:8px; }
  #panel h1 { font-size:15px; margin:0; }
  #panel button { border:0; background:none; color:var(--muted); cursor:pointer; font:inherit; padding:0; }
  #panel .sub { color:var(--muted); margin:4px 0 10px; }
  #panel h2 { font-size:11px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); margin:12px 0 6px; }
  #panel ul { margin:0; padding-left:0; list-style:none; }
  #panel li { margin:3px 0; }
  #panel li a { color:var(--chg); cursor:pointer; text-decoration:none; font-weight:600; }
  #panel li a:hover { text-decoration:underline; }
  .chip { display:inline-block; width:12px; height:12px; border-radius:3px; vertical-align:-1px; margin-right:6px; border:1.5px solid; }
  .legend { display:grid; grid-template-columns:1fr 1fr; gap:4px 8px; }
  .hint { color:var(--muted); margin-top:12px; line-height:1.5; }
  .ovr { margin-top:6px; padding:5px 8px; background:#f5f3ff; border-radius:6px; color:#4c1d95; }
  .hint a { color:var(--ink); cursor:pointer; text-decoration:underline dotted; }
  #err { position:fixed; bottom:12px; left:12px; right:12px; color:#b91c1c; white-space:pre-wrap; font:12px monospace; }
  #mini { position:fixed; right:22px; bottom:22px; background:rgba(255,255,255,.94); border:1px solid var(--line);
          border-radius:10px; box-shadow:0 4px 18px rgba(15,23,42,.10); padding:8px; cursor:crosshair; overflow:hidden; }
  #mini.off { display:none; }
  #mini .frame { position:relative; }
  #mini svg { display:block; }
  #mini svg * { vector-effect:non-scaling-stroke; }
  #mini svg path { marker-start:none !important; marker-end:none !important; }
  #mini .glab { position:absolute; font-size:10px; font-weight:650; color:#334155; line-height:1.15; white-space:nowrap;
                margin:3px; padding:1px 4px; background:rgba(255,255,255,.85); border-radius:4px; cursor:zoom-in; }
  #mini .glab:hover { background:#fff; color:#2563eb; }
  #mini .view { position:absolute; border:1.5px solid #2563eb; background:rgba(37,99,235,.10); border-radius:2px; pointer-events:none; }
  .bar { position:fixed; z-index:2; }
  #hbar { left:12px; right:22px; bottom:2px; height:12px; }
  #vbar { top:12px; bottom:22px; right:2px; width:12px; }
  .thumb { position:absolute; background:rgba(100,116,139,.32); border-radius:6px; }
  #hbar .thumb { top:3px; height:6px; }
  #vbar .thumb { left:3px; width:6px; }
  .bar:hover .thumb, .thumb.active { background:rgba(71,85,105,.62); }
  #hbar:hover .thumb, #hbar .thumb.active { top:1px; height:10px; }
  #vbar:hover .thumb, #vbar .thumb.active { left:1px; width:10px; }
</style></head>
<body>
<div id="stage"></div>
<aside id="panel"></aside>
<div id="mini"><div class="frame"></div></div>
<div id="hbar" class="bar"><div class="thumb"></div></div>
<div id="vbar" class="bar"><div class="thumb"></div></div>
<div id="err"></div>
<script id="data" type="application/json">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById('data').textContent);

// TeX between single dollars -> MathML, before layout, so boxes fit the typeset math.
function typeset(src) {
  return src.replace(/\$([^$]+)\$/g, (_, tex) => {
    let m = katex.renderToString(tex, {output: 'mathml', throwOnError: false});
    return m.replace(/<annotation[^>]*>[\s\S]*?<\/annotation>/g, '')
            .replace(/&#x([0-9a-f]+);/gi, (_, h) => String.fromCodePoint(parseInt(h, 16)))
            .replace(/&#(\d+);/g, (_, d) => String.fromCodePoint(+d))
            .replace(/"/g, "'").replace(/\n/g, '');
  });
}

function buildPanel(p) {
  const el = document.getElementById('panel');
  const legend = p.legend.map(([k, fill, stroke, dashed]) =>
    `<span><span class="chip" style="background:${fill};border-color:${stroke};border-style:${dashed ? 'dashed' : 'solid'}"></span>${k}</span>`).join('');
  const changed = p.baseline === null ? '<li>no baseline: nothing highlighted</li>'
    : (p.changed.length ? p.changed.map(c => `<li><a data-id="${c.id}">${typeset(c.title)}</a></li>`).join('') : '<li>none</li>');
  const removed = p.removed.length ? `<h2>Removed</h2><ul>${p.removed.map(t => `<li>${typeset(t)}</li>`).join('')}</ul>` : '';
  el.innerHTML = `<header><h1>${p.title}</h1><button id="fold">hide</button></header>
    <div class="sub">${p.subtitle}</div>
    <div>source <b>${p.commit}</b> · ${p.audited}<br>rendered ${p.rendered}</div>
    ${p.overrides.length ? `<div class="ovr">committed mk_dev plus: <b>${p.overrides.join('</b>, <b>')}</b></div>` : ''}
    <h2>${p.baseline === null ? 'Changed since baseline' : p.baseline}</h2><ul>${changed}</ul>${removed}
    <h2>Node kinds</h2><div class="legend">${legend}</div>
    <div class="hint"><span id="wheelhint"></span> · <a id="wheelmode"></a><br>
      drag (left or middle): pan · arrows: pan · +/−: zoom<br>
      zoomed out, boxes show titles only<br>
      double-click a box or a stage: zoom to it<br>
      F: fit · M: map · click the map to go there,<br>a stage name on it to zoom to the stage<br>
      click a changed item to fly to it</div>`;
  document.getElementById('fold').onclick = () => {
    el.classList.toggle('collapsed');
    document.getElementById('fold').textContent = el.classList.contains('collapsed') ? 'show' : 'hide';
  };
  document.getElementById('wheelmode').onclick = () => setWheelMode(wheelMode === 'pan' ? 'zoom' : 'pan');
  setWheelMode(wheelMode);
}

// What a plain wheel event does. 'pan' suits scroll devices that emit wheel events
// with both deltas (TrackPoint middle-button scrolling, touchpads); 'zoom' suits a
// mouse wheel. Ctrl+wheel (and a touchpad pinch, which arrives as one) always zooms.
let wheelMode = 'pan';
try { wheelMode = localStorage.getItem('workvis.wheel') || 'pan'; } catch (e) {}
function setWheelMode(m) {
  wheelMode = m;
  try { localStorage.setItem('workvis.wheel', m); } catch (e) {}
  document.getElementById('wheelhint').textContent = m === 'pan'
    ? 'wheel / TrackPoint scroll: pan · Ctrl+wheel: zoom' : 'wheel: zoom · Shift+wheel or sideways scroll: pan';
  document.getElementById('wheelmode').textContent = m === 'pan' ? 'make wheel zoom' : 'make wheel pan';
}

function panZoom(svg) {
  const stage = document.getElementById('stage');
  const bb = svg.viewBox.baseVal;
  const home = {x: bb.x, y: bb.y, w: bb.width, h: bb.height};
  const vb = {...home};
  const listeners = [];
  const apply = () => { svg.setAttribute('viewBox', `${vb.x} ${vb.y} ${vb.w} ${vb.h}`); listeners.forEach(f => f()); };
  const toSvg = (cx, cy, ctm) => { const p = svg.createSVGPoint(); p.x = cx; p.y = cy; return p.matrixTransform((ctm || svg.getScreenCTM()).inverse()); };
  const zoomAt = (cx, cy, k) => { const p = toSvg(cx, cy); vb.x = p.x - (p.x - vb.x) * k; vb.y = p.y - (p.y - vb.y) * k; vb.w *= k; vb.h *= k; apply(); };
  const fit = () => { Object.assign(vb, home); apply(); };
  const animateTo = t => {
    const s0 = {...vb}, t0 = performance.now();
    const step = now => {
      const u = Math.min(1, (now - t0) / 280), e = 1 - Math.pow(1 - u, 3);
      for (const k of ['x', 'y', 'w', 'h']) vb[k] = s0[k] + (t[k] - s0[k]) * e;
      apply(); if (u < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  };
  // Zoom so box b fills frac of the screen, but never past kmax screen px per unit.
  const fitBox = (b, frac, kmax) => {
    const W = stage.clientWidth, H = stage.clientHeight;
    const k = Math.min(frac * W / b.w, frac * H / b.h, kmax || Infinity), w = W / k, h = H / k;
    animateTo({x: b.x + b.w / 2 - w / 2, y: b.y + b.h / 2 - h / 2, w, h});
  };
  const nodeBox = g => boxOf(svg, g.querySelector('.label-container') || g);
  // The diagram region actually on screen: the viewBox is letterboxed into the stage.
  const visible = () => { const r = stage.getBoundingClientRect(), a = toSvg(r.left, r.top), b = toSvg(r.right, r.bottom); return {x: a.x, y: a.y, w: b.x - a.x, h: b.y - a.y}; };
  const shift = (dx, dy) => { vb.x += dx; vb.y += dy; apply(); };  // diagram units
  const panBy = (dx, dy) => { const m = svg.getScreenCTM(); shift(dx / m.a, dy / m.d); };  // screen px
  const centerOn = (x, y) => { const v = visible(); shift(x - (v.x + v.w / 2), y - (v.y + v.h / 2)); };
  const px = (e, d) => d * (e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? stage.clientHeight : 1);
  stage.addEventListener('wheel', e => {
    e.preventDefault();
    let dx = px(e, e.deltaX), dy = px(e, e.deltaY);
    if (e.ctrlKey || e.metaKey || (wheelMode === 'zoom' && !e.shiftKey && dx === 0)) { zoomAt(e.clientX, e.clientY, Math.exp(dy * 0.0015)); return; }
    if (e.shiftKey && dx === 0) { dx = dy; dy = 0; }
    panBy(dx, dy);
  }, {passive: false});
  window.addEventListener('resize', apply);
  let drag = null;
  stage.addEventListener('mousedown', e => {
    if (e.button !== 0 && e.button !== 1) return;
    e.preventDefault();  // a middle press would otherwise start autoscroll
    drag = {ctm: svg.getScreenCTM(), p0: toSvg(e.clientX, e.clientY), vb0: {...vb}};
    stage.classList.add('dragging');
  });
  window.addEventListener('mousemove', e => {
    if (!drag) return;
    const p = toSvg(e.clientX, e.clientY, drag.ctm);
    vb.x = drag.vb0.x - (p.x - drag.p0.x); vb.y = drag.vb0.y - (p.y - drag.p0.y); apply();
  });
  window.addEventListener('mouseup', () => { drag = null; stage.classList.remove('dragging'); });
  stage.addEventListener('auxclick', e => e.preventDefault());
  const arrows = {ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1]};
  window.addEventListener('keydown', e => {
    if (e.key === 'f' || e.key === 'F' || e.key === '0') fit();
    if (e.key === 'm' || e.key === 'M') document.getElementById('mini').classList.toggle('off');
    const r = stage.getBoundingClientRect();
    if (e.key === '+' || e.key === '=') zoomAt(r.width / 2, r.height / 2, 0.8);
    if (e.key === '-') zoomAt(r.width / 2, r.height / 2, 1.25);
    if (arrows[e.key]) { e.preventDefault(); panBy(arrows[e.key][0] * r.width * 0.15, arrows[e.key][1] * r.height * 0.15); }
  });
  document.getElementById('panel').addEventListener('click', e => {
    const a = e.target.closest && e.target.closest('a[data-id]');
    if (!a) return;
    const g = svg.querySelector(`g.node[id*="-n_${a.dataset.id}-"]`);
    if (g) fitBox(nodeBox(g), 0.6, 1.4);
  });
  // Double-click: a box zooms to read it, empty space in a stage or loop zooms to that.
  stage.addEventListener('dblclick', e => {
    const t = e.target.closest && (e.target.closest('g.node') || e.target.closest('[data-node]'));
    const g = t && (t.matches('g.node') ? t : svg.querySelector(`[id="${t.dataset.node}"]`));
    if (g) return fitBox(nodeBox(g), 0.6, 1.4);
    const p = toSvg(e.clientX, e.clientY);
    const hit = clustersOf(svg).filter(c => p.x >= c.b.x && p.x <= c.b.x + c.b.w && p.y >= c.b.y && p.y <= c.b.y + c.b.h)
      .sort((a, b) => a.b.w * a.b.h - b.b.w * b.b.h)[0];
    hit ? fitBox(hit.b, 0.96) : animateTo(home);
  });
  return {home, visible, shift, centerOn, onView: f => listeners.push(f), fit, fitBox};
}

// Box of an element in the diagram's own units, whatever the current zoom.
function boxOf(svg, el) {
  const b = el.getBBox(), m = svg.getCTM().inverse().multiply(el.getCTM());
  return {x: m.e + b.x * m.a, y: m.f + b.y * m.d, w: b.width * m.a, h: b.height * m.d};
}

// Stage and loop boxes, each with its label and whether it sits inside another.
function clustersOf(svg) {
  const cs = [...svg.querySelectorAll('g.cluster')].map(c => ({r: c.querySelector('rect'), lab: c.querySelector('.cluster-label')}))
    .filter(c => c.r && c.lab).map(c => ({...c, b: boxOf(svg, c.r), text: c.lab.textContent.trim()}));
  const inside = (a, o) => a !== o && a.b.x >= o.b.x && a.b.y >= o.b.y && a.b.x + a.b.w <= o.b.x + o.b.w && a.b.y + a.b.h <= o.b.y + o.b.h;
  cs.forEach(c => { c.nested = cs.some(o => inside(c, o)); });
  return cs;
}

// Inset of the whole diagram: shapes and edges without text, stage labels on top,
// changed boxes filled orange, and the on-screen region as a blue frame.
function minimap(svg, view) {
  const box = document.getElementById('mini'), frame = box.querySelector('.frame');
  const h = view.home, s = Math.min(230 / h.w, 340 / h.h);
  const mini = svg.cloneNode(true);  // keeps the id, so Mermaid's scoped styles still apply
  mini.querySelectorAll('foreignObject, .edgeLabels').forEach(n => n.remove());
  // Mermaid writes class styles inline with !important, so override them inline.
  mini.querySelectorAll('.node .label-container').forEach(n => {
    n.style.setProperty('stroke-width', '1px', 'important');
    if (n.closest('.node').classList.contains('changed')) {
      n.style.setProperty('fill', '#fdba74', 'important'); n.style.setProperty('stroke', '#ea580c', 'important');
    }
  });
  mini.setAttribute('viewBox', `${h.x} ${h.y} ${h.w} ${h.h}`);
  mini.setAttribute('width', h.w * s); mini.setAttribute('height', h.h * s);
  frame.appendChild(mini);
  // Label the stages only: a loop nested in a stage would sit on its stage's label.
  clustersOf(svg).filter(c => !c.nested).forEach(({b, text}) => {
    const d = document.createElement('div');
    d.className = 'glab'; d.textContent = text;
    Object.assign(d.style, {left: (b.x - h.x) * s + 'px', top: (b.y - h.y) * s + 'px'});
    d.addEventListener('mousedown', e => { e.preventDefault(); e.stopPropagation(); view.fitBox(b, 0.96); });
    frame.appendChild(d);
  });
  const vr = document.createElement('div'); vr.className = 'view'; frame.appendChild(vr);
  view.onView(() => {
    const v = view.visible();
    Object.assign(vr.style, {left: (v.x - h.x) * s + 'px', top: (v.y - h.y) * s + 'px', width: v.w * s + 'px', height: v.h * s + 'px'});
  });
  const at = e => { const r = mini.getBoundingClientRect(); return [h.x + (e.clientX - r.left) / s, h.y + (e.clientY - r.top) / s]; };
  let dragging = false;
  box.addEventListener('mousedown', e => { e.preventDefault(); dragging = true; view.centerOn(...at(e)); });
  window.addEventListener('mousemove', e => { if (dragging) view.centerOn(...at(e)); });
  window.addEventListener('mouseup', () => { dragging = false; });
  box.addEventListener('wheel', e => e.preventDefault(), {passive: false});
}

// Semantic zoom. Once the detail lines would be too small to read, each box shows
// only its title, kept near TITLE_PX on screen (capped so it fits the box), and each
// stage and loop shows its name in large faint letters behind its boxes.
const LINE_UNITS = 14, OVER_BELOW = 5.6, DETAIL_ABOVE = 6.4, TITLE_PX = 15, MARK_PX = 30, LOOP_MARK_PX = 19;
function semanticZoom(svg, view) {
  const NS = 'http://www.w3.org/2000/svg', root = svg.querySelector('g.root');
  const titles = [], marks = [];
  const layer = document.createElementNS(NS, 'g'); layer.setAttribute('class', 'lod-titles');
  svg.querySelectorAll('g.node').forEach(g => {
    const t = g.querySelector('.t'), shape = g.querySelector('.label-container') || g;
    if (!t) return;
    const b = boxOf(svg, shape);
    const fo = document.createElementNS(NS, 'foreignObject');
    for (const [k, v] of Object.entries({x: b.x, y: b.y, width: b.w, height: b.h})) fo.setAttribute(k, v);
    fo.dataset.node = g.id;
    const wrap = document.createElement('div'), d = document.createElement('div');
    wrap.className = 'lodt' + (g.classList.contains('changed') ? ' changed' : '');
    d.innerHTML = t.innerHTML;
    wrap.appendChild(d); fo.appendChild(wrap); layer.appendChild(fo);
    titles.push({d, b});
  });
  root.appendChild(layer);
  const back = document.createElementNS(NS, 'g'); back.setAttribute('class', 'lod-marks');
  clustersOf(svg).forEach(c => {
    const t = document.createElementNS(NS, 'text');
    t.textContent = c.text; t.setAttribute('dominant-baseline', 'hanging');
    back.appendChild(t); marks.push({t, c});
  });
  root.insertBefore(back, root.querySelector('g.edgePaths'));  // above stage fills, below edges and boxes

  // Largest title that fits each box, and the widest each stage name may run, in diagram units.
  svg.classList.add('lod-over');
  for (const it of titles) {
    let lo = 4, hi = Math.max(8, it.b.h);
    for (let i = 0; i < 14; i++) {
      const mid = (lo + hi) / 2; it.d.style.fontSize = mid + 'px';
      (it.d.offsetHeight <= it.b.h * 0.86 && it.d.scrollWidth <= it.d.clientWidth + 1) ? lo = mid : hi = mid;
    }
    it.max = lo;
  }
  for (const m of marks) {
    m.t.setAttribute('font-size', 100);
    m.max = 100 * (m.c.b.w * 0.9) / Math.max(1, m.t.getComputedTextLength());
  }
  let over = null;
  view.onView(() => {
    const k = svg.getScreenCTM().a, line = LINE_UNITS * k;
    const now = over === null ? line < (OVER_BELOW + DETAIL_ABOVE) / 2 : (over ? line < DETAIL_ABOVE : line < OVER_BELOW);
    if (now !== over) { over = now; svg.classList.toggle('lod-over', over); }
    if (!over) return;
    for (const it of titles) it.d.style.fontSize = Math.min(TITLE_PX / k, it.max) + 'px';
    for (const m of marks) {
      const fs = Math.min((m.c.nested ? LOOP_MARK_PX : MARK_PX) / k, m.max), pad = 12 / k;
      m.t.setAttribute('font-size', fs);
      m.t.setAttribute('x', m.c.b.x + pad); m.t.setAttribute('y', m.c.b.y + pad);
    }
  });
}

// Scrollbars over the union of the diagram and the on-screen region.
function scrollbars(view) {
  const axes = [{bar: 'hbar', p: 'x', d: 'w', pos: 'left', size: 'width', c: 'clientX'},
                {bar: 'vbar', p: 'y', d: 'h', pos: 'top', size: 'height', c: 'clientY'}];
  for (const a of axes) {
    a.el = document.getElementById(a.bar); a.th = a.el.firstElementChild;
    a.span = () => { const v = view.visible(), h = view.home;
      return [Math.min(h[a.p], v[a.p]), Math.max(h[a.p] + h[a.d], v[a.p] + v[a.d]), v]; };
    a.th.addEventListener('mousedown', e => {
      e.preventDefault(); e.stopPropagation();
      const [lo, hi] = a.span();
      a.drag = {c0: e[a.c], k: (hi - lo) / a.el.getBoundingClientRect()[a.size], moved: 0};
      a.th.classList.add('active');
    });
    a.el.addEventListener('mousedown', e => {  // a press on the track centres the view there
      e.preventDefault();
      const [lo, hi, v] = a.span(), r = a.el.getBoundingClientRect();
      const t = lo + (e[a.c] - r[a.pos]) / r[a.size] * (hi - lo);
      a.p === 'x' ? view.shift(t - (v.x + v.w / 2), 0) : view.shift(0, t - (v.y + v.h / 2));
    });
  }
  window.addEventListener('mousemove', e => {
    for (const a of axes) {
      if (!a.drag) continue;
      const step = (e[a.c] - a.drag.c0) * a.drag.k - a.drag.moved;
      a.drag.moved += step;
      a.p === 'x' ? view.shift(step, 0) : view.shift(0, step);
    }
  });
  window.addEventListener('mouseup', () => axes.forEach(a => { a.drag = null; a.th.classList.remove('active'); }));
  view.onView(() => {
    for (const a of axes) {
      const [lo, hi, v] = a.span(), L = a.el.getBoundingClientRect()[a.size];
      const len = Math.max(28, v[a.d] / (hi - lo) * L);
      a.th.style[a.size] = len + 'px';
      a.th.style[a.pos] = Math.min(L - len, (v[a.p] - lo) / (hi - lo) * L) + 'px';
    }
  });
}

(async () => {
  buildPanel(DATA.panel);
  mermaid.initialize({
    startOnLoad: false, securityLevel: 'loose', theme: 'base',
    maxTextSize: 5000000,  // typeset MathML inflates the source far past the 50k default
    themeVariables: {fontFamily: 'Inter, system-ui, sans-serif', fontSize: '15px',
                     lineColor: '#64748b', primaryTextColor: '#0f172a',
                     clusterBkg: '#ffffff', clusterBorder: '#e2e8f0'},
    flowchart: {htmlLabels: true, useMaxWidth: false, curve: 'basis',
                nodeSpacing: 46, rankSpacing: 58, padding: 18, wrappingWidth: 1200},
  });
  try {
    await document.fonts.ready;
    const {svg} = await mermaid.render('wv', typeset(DATA.mermaid));
    const stage = document.getElementById('stage');
    stage.innerHTML = svg;
    const el = stage.querySelector('svg');
    el.removeAttribute('style'); el.removeAttribute('width'); el.removeAttribute('height');
    const view = panZoom(el);
    minimap(el, view);  // clones the drawing, so it runs before the overview layers are added
    semanticZoom(el, view);
    scrollbars(view);
    view.fit();
    const nodes = el.querySelectorAll('g.node').length, mathErrors = el.querySelectorAll('merror, .katex-error').length;
    window.WORKVIS_CHECK = {ok: nodes === DATA.n_nodes && mathErrors === 0, nodes, expected: DATA.n_nodes, mathErrors, error: ''};
  } catch (e) {
    document.getElementById('err').textContent = 'render failed: ' + (e && e.message || e);
    window.WORKVIS_CHECK = {ok: false, nodes: 0, expected: DATA.n_nodes, mathErrors: null, error: String(e && e.message || e)};
  }
})();
</script>
</body></html>
"""


if __name__ == '__main__':
    sys.exit(main())
