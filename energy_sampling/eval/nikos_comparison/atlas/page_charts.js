/* Plotly rendering of the atlas figures.
 *
 * Plotly does not read CSS custom properties, so every colour is pulled from the
 * stylesheet at draw time via getComputedStyle and the whole set is redrawn when the
 * theme changes. That keeps ONE definition of the palette -- the :root blocks -- and
 * avoids a second, silently diverging copy in JS.
 *
 * Encoding, held constant across all four basin panels so the same point means the
 * same thing everywhere:
 *     colour  = basin minimum energy   (sequential, deepest = strongest)
 *     area    = times the search found it
 *     symbol  = packing family, gamma circle / beta square
 * Family is carried by SHAPE, not colour, because colour is already spent on energy;
 * the legend names both shapes so identity is never colour-alone.
 */
const D = window.__ATLAS__;

const css = (n) => getComputedStyle(document.documentElement)
  .getPropertyValue(n).trim();

function theme() {
  return {
    surface: css('--surface'), card: css('--card'), grid: css('--grid'),
    ink: css('--ink'), ink2: css('--ink-2'), ink3: css('--ink-3'),
    known: css('--known'), nik: css('--nik'),
    ramp: ['--s1', '--s2', '--s3', '--s4', '--s5', '--s6'].map(css),
  };
}

const FONT = '"IBM Plex Sans", system-ui, sans-serif';
const MONO = '"IBM Plex Mono", monospace';
const CONF = { displayModeBar: false, responsive: true };

/* Deepest energy -> the ramp's far end. Energy is negative and "more" means "more
 * negative", so the scale is built reversed rather than fighting cmin/cmax. */
function escale(t) {
  const r = t.ramp;
  return r.map((c, i) => [i / (r.length - 1), r[r.length - 1 - i]]);
}

/* Named structures cluster tightly -- four of the five relax into a 0.8 kJ/mol,
 * 0.02-packing-coefficient neighbourhood -- so free-floating text marks collide no
 * matter which side they are placed on. Plotly does no collision avoidance, so these
 * become arrowed callouts with a deterministic vertical stagger: the label is pushed
 * far enough out to stand clear and an arrow keeps it tied to its point. */
const STAGGER = [-34, -62, -90, 34, 62];
function callouts(items, t) {
  return items.slice()
    .sort((a, b) => (b.y - a.y) || (a.x - b.x))
    .map((o, i) => ({
      x: o.x, y: o.y, text: o.text, showarrow: true,
      ax: 0, ay: STAGGER[i % STAGGER.length],
      arrowhead: 0, arrowwidth: 1, arrowcolor: o.color || t.ink3, standoff: 5,
      font: { family: MONO, size: 10, color: o.color || t.ink2 },
      bgcolor: t.card, borderpad: 2, opacity: 0.97,
    }));
}

function layout(t, xt, yt, extra) {
  return Object.assign({
    paper_bgcolor: t.card, plot_bgcolor: t.card,
    font: { family: FONT, size: 12, color: t.ink2 },
    margin: { l: 68, r: 24, t: 16, b: 56 },
    hoverlabel: { font: { family: MONO, size: 12 }, bgcolor: t.card,
                  bordercolor: t.grid, align: 'left' },
    xaxis: { title: { text: xt, font: { size: 12.5, color: t.ink2 } },
             gridcolor: t.grid, zeroline: false, linecolor: t.grid,
             ticks: 'outside', tickcolor: t.grid, tickfont: { color: t.ink3, size: 11 } },
    yaxis: { title: { text: yt, font: { size: 12.5, color: t.ink2 } },
             gridcolor: t.grid, zeroline: false, linecolor: t.grid,
             ticks: 'outside', tickcolor: t.grid, tickfont: { color: t.ink3, size: 11 } },
    legend: { orientation: 'h', y: 1.02, yanchor: 'bottom', x: 0,
              font: { size: 11.5, color: t.ink2 } },
  }, extra || {});
}

const NMAX = Math.max(...D.basins.map(b => b.n));

function hoverText(b) {
  const held = b.holds.length
    ? '<br>holds: <b>' + b.holds.join(' / ') + '</b>' : '';
  return `<b>basin ${b.basin}</b><br>${b.label}`
    + `<br>found ${b.n.toLocaleString()}x  \u00b7  min E `
    + `${b.emin.toFixed(2)} kJ/mol`
    + `<br>diameter ${b.diam.toFixed(4)}` + held;
}

/* One trace per family so the legend entries are real series, not annotations. */
function basinTraces(t, xk, yk) {
  return ['gamma', 'beta'].map((fam) => {
    const S = D.basins.filter(b => b.fam === fam);
    return {
      type: 'scatter', mode: 'markers',
      name: fam === 'gamma' ? '\u03b3  stack + herringbone'
                             : '\u03b2  stacked layers',
      x: S.map(b => b[xk]), y: S.map(b => b[yk]),
      text: S.map(hoverText), hovertemplate: '%{text}<extra></extra>',
      marker: {
        symbol: fam === 'gamma' ? 'circle' : 'square',
        size: S.map(b => b.n), sizemode: 'area',
        sizeref: (2.0 * NMAX) / (34 ** 2), sizemin: 5,
        color: S.map(b => b.emin), colorscale: escale(t),
        cmin: Math.min(...D.basins.map(b => b.emin)),
        cmax: Math.max(...D.basins.map(b => b.emin)),
        line: { color: t.card, width: 1.5 },
        colorbar: fam === 'gamma' ? {
          title: { text: 'min E<br>kJ/mol', font: { size: 11, color: t.ink3 } },
          thickness: 11, len: 0.62, y: 0.5, outlinewidth: 0,
          tickfont: { size: 10, color: t.ink3, family: MONO },
        } : undefined,
        showscale: fam === 'gamma',
      },
    };
  });
}

/* Basins holding a named structure get a ring plus a direct label -- the relief the
 * palette check requires, and the thing that makes the deep/named ones findable
 * without hovering every point. */
function heldOverlay(t, xk, yk) {
  const S = D.basins.filter(b => b.holds.length);
  return {
    type: 'scatter', mode: 'markers', showlegend: false,
    x: S.map(b => b[xk]), y: S.map(b => b[yk]),
    hoverinfo: 'skip',
    marker: { symbol: 'circle-open', size: 26, color: t.ink3, line: { width: 1.4 } },
  };
}

function drawBasins(id, xk, yk, xt, yt, t, extra) {
  //: D.basins[].holds already excludes non-members -- a structure further from its
  //: nearest member than the 0.10 cut is not IN that basin, and labelling the basin
  //: with its name put that name at the BASIN's energy while the landscape panel
  //: drew the structure at its own. Same object, two different places.
  const held = D.basins.filter(b => b.holds.length)
    .map(b => ({ x: b[xk], y: b[yk], text: b.holds.join(' = ') }));
  const L = layout(t, xt, yt, extra);
  L.annotations = callouts(held, t);
  L.margin = Object.assign({}, L.margin, { t: 34 });
  (D.unplaced || []).forEach((u) => {
    L.margin = Object.assign({}, L.margin, { b: 92 });
    L.annotations.push({
      xref: 'paper', yref: 'paper', x: 1, y: -0.155,
      xanchor: 'right', yanchor: 'top', showarrow: false,
      text: `${u.name} matches no basin at the 0.10 cut `
          + `(nearest member ${u.rdf.toFixed(2)} away) and is not drawn`,
      font: { family: MONO, size: 9.5, color: t.ink3 },
    });
  });
  Plotly.react(id, [...basinTraces(t, xk, yk), heldOverlay(t, xk, yk)], L, CONF);
}

function drawLandscape(t) {
  const dens = {
    type: 'heatmap', x: D.dens.x, y: D.dens.y, z: D.dens.z,
    colorscale: t.ramp.map((c, i) => [i / (t.ramp.length - 1), c]),
    zsmooth: false, showscale: true, hoverongaps: false,
    colorbar: { title: { text: 'structures<br>per cell', font: { size: 11, color: t.ink3 } },
                thickness: 11, len: 0.62, outlinewidth: 0,
                tickfont: { size: 10, color: t.ink3, family: MONO } },
    hovertemplate: 'packing coeff %{x:.3f}<br>energy %{y:.1f} kJ/mol'
                 + '<br><b>%{z}</b> structures<extra></extra>',
  };
  const YLO = -64.2, YHI = -28.5;
  const note = [];
  const traces = [dens];
  const seen = {};
  //: co-located structures share one marker and one label -- nik00000 relaxes onto
  //: ACRDIN12 exactly, and drawing that as two labelled points implies two basins.
  const key = (s) => s.pc1.toFixed(3) + '|' + s.e1.toFixed(2);
  const merged = {};
  D.named.forEach(s => (merged[key(s)] = merged[key(s)] || []).push(s.name));
  const drawn = {};
  D.named.slice().sort((a, b) => a.e1 - b.e1).forEach((s, i) => {
    const col = s.kind === 'known' ? t.known : t.nik;
    const lab = s.kind === 'known' ? 'experimental (CSD)' : "Nikos' proposals";
    const off = s.e0 > YHI;                       // starts above the panel
    const y0 = off ? YHI : s.e0;
    traces.push({
      type: 'scatter', mode: 'lines', x: [s.pc0, s.pc1], y: [y0, s.e1],
      line: { color: col, width: 1.4, dash: 'dot' },
      hoverinfo: 'skip', showlegend: false,
    });
    traces.push({                                   // as supplied
      type: 'scatter', mode: off ? 'markers+text' : 'markers',
      x: [s.pc0], y: [y0],
      marker: { symbol: off ? 'triangle-up-open' : 'circle-open',
                size: off ? 12 : 10, color: col, line: { width: 1.8 } },
      text: off ? [`${s.name} unrelaxed ${s.e0.toFixed(1)} \u2191`] : undefined,
      textposition: 'bottom center',
      textfont: { family: MONO, size: 10, color: col },
      name: lab, legendgroup: lab, showlegend: !seen[lab],
      hovertemplate: `<b>${s.name}</b> as supplied<br>packing coeff `
        + `${s.pc0.toFixed(3)}<br>energy ${s.e0.toFixed(2)} kJ/mol`
        + (off ? '<br><i>above the panel</i>' : '') + `<extra></extra>`,
    });
    seen[lab] = true;
    const names = merged[key(s)];
    const first = !drawn[key(s)];
    drawn[key(s)] = true;
    if (first) note.push({ x: s.pc1, y: s.e1, text: names.join(' = '), color: col });
    traces.push({                                   // after relaxation
      type: 'scatter', mode: 'markers', x: [s.pc1], y: [s.e1],
      marker: { symbol: 'circle', size: 11, color: col,
                line: { color: t.card, width: 1.5 } },
      showlegend: false, legendgroup: lab,
      hovertemplate: `<b>${s.name}</b> relaxed<br>packing coeff `
        + `${s.pc1.toFixed(3)}<br>energy ${s.e1.toFixed(2)} kJ/mol<extra></extra>`,
    });
  });
  const L = layout(t, 'packing coefficient', 'MACE lattice energy  (kJ/mol)', {
    margin: { l: 74, r: 24, t: 16, b: 56 },
    xaxis: Object.assign(layout(t, '', '').xaxis,
      { title: { text: 'packing coefficient', font: { size: 12.5, color: t.ink2 } },
        range: [0.55, 0.80] }),
    yaxis: Object.assign(layout(t, '', '').yaxis,
      { title: { text: 'MACE lattice energy  (kJ/mol)',
                 font: { size: 12.5, color: t.ink2 } },
        range: [YLO, YHI] }),
    shapes: [{ type: 'line', xref: 'paper', x0: 0, x1: 1,
               y0: D.meta.floor, y1: D.meta.floor,
               line: { color: t.ink3, width: 1, dash: 'dash' } }],
    annotations: [{ xref: 'paper', x: 0.012, y: D.meta.floor, xanchor: 'left',
                    yanchor: 'bottom', text: `energy floor ${D.meta.floor}`,
                    showarrow: false,
                    font: { family: MONO, size: 10, color: t.ink3 } },
                  ...callouts(note, t)],
  });
  Plotly.react('p-land', traces, L, CONF);
}

function drawAll() {
  const t = theme();
  drawLandscape(t);
  drawBasins('p-pce', 'pc', 'emin', 'packing coefficient',
             'basin minimum energy  (kJ/mol)', t);
  drawBasins('p-bt', 'beta', 'theta_std', 'monoclinic \u03b2 angle  (deg)',
             'spread of interplane angles  (deg)', t);
  drawBasins('p-mds', 'mds0', 'mds1', 'RDF embedding dim 1',
             'dim 2', t);
  drawBasins('p-nd', 'n_nbr', 'mean_d', 'coordination number',
             'mean neighbour distance  (\u00c5)', t,
             { xaxis: { title: { text: 'coordination number',
                                 font: { size: 12.5, color: t.ink2 } },
                        dtick: 1, gridcolor: t.grid, zeroline: false,
                        linecolor: t.grid, ticks: 'outside', tickcolor: t.grid,
                        tickfont: { color: t.ink3, size: 11 } } });
}

function fillMeta() {
  document.getElementById('m-basins').textContent = D.meta.n_basins;
  document.getElementById('m-struct').textContent = D.meta.n_struct.toLocaleString();
  document.getElementById('m-link').textContent = `${D.meta.linkage}, cut ${D.meta.cut}`;
  document.getElementById('m-floor').textContent = `${D.meta.floor} kJ/mol`;

  const pct = (100 * D.dens.n_shown / D.dens.n_total).toFixed(1);
  document.getElementById('c-land').innerHTML =
    `<b>${D.dens.n_total.toLocaleString()} physical structures</b>, of which `
    + `${D.dens.n_shown.toLocaleString()} (${pct}%) fall inside these axes \u2014 the rest `
    + `sit above &minus;30&nbsp;kJ/mol or outside the density window and are not drawn. `
    + `Neither known form is the minimum: they sit 2.7 and 4.2&nbsp;kJ/mol above the `
    + `floor. Nikos' structures travel 6&ndash;39&nbsp;kJ/mol on relaxation because they `
    + `arrive unrelaxed; the polymorphs travel 1&ndash;1.5.`;

  const tb = document.querySelector('#tbl tbody');
  tb.innerHTML = D.basins.map(b => {
    const held = b.holds.map((h, i) => {
      const kind = h.startsWith('nik') ? 'tag n' : 'tag';
      const far = b.holds_rdf[i] > 0.10
        ? ` <span style="opacity:.7">(${b.holds_rdf[i].toFixed(2)} away)</span>`
        : '';
      return `<span class="${kind}">${h}</span>${far}`;
    }).join(' ');
    return `<tr><td>${b.basin}</td><td class="r">${b.n.toLocaleString()}</td>`
      + `<td class="r">${b.emin.toFixed(2)}</td>`
      + `<td>${b.fam === 'gamma' ? '\u03b3' : '\u03b2'}</td>`
      + `<td class="r">${b.beta.toFixed(1)}</td><td class="r">${b.theta_std.toFixed(1)}</td>`
      + `<td class="r">${b.stack === null ? '\u2014' : b.stack.toFixed(2)}</td>`
      + `<td class="r">${b.n_nbr}</td><td class="r">${b.diam.toFixed(4)}</td>`
      + `<td>${held || '\u2014'}</td></tr>`;
  }).join('');
}

fillMeta();
drawAll();

/* Redraw on either route to a theme change: the OS setting, and the viewer's
 * explicit toggle (which stamps data-theme on <html>). */
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', drawAll);
new MutationObserver(drawAll).observe(document.documentElement,
  { attributes: true, attributeFilter: ['data-theme'] });
