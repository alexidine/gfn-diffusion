"""
Assemble the Plotly version of the atlas figures into one self-contained page.

An artifact's CSP blocks external scripts, so plotly.min.js is INLINED from the
installed plotly package rather than pulled from a CDN. That is ~4.6 MB, well inside
the 16 MB page cap, and it buys real hover -- which is the point of this version:
the hand-drawn SVG figures could not label 42 basins without the labels colliding.

Inputs are the same intermediates the SVG renderers read, so the two versions cannot
drift: plotly_data.json comes from export_plotly_data.py, which reads
coarse_landscape.json -- THE basin definition.
"""
import json
import os

import plotly

SP = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(SP, 'acridine_basin_figures.html')

JS = os.path.join(os.path.dirname(plotly.__file__), 'package_data', 'plotly.min.js')
lib = open(JS, encoding='utf-8').read()
tpl = open(os.path.join(SP, 'page_template.html'), encoding='utf-8').read()
charts = open(os.path.join(SP, 'page_charts.js'), encoding='utf-8').read()
data = json.load(open(os.path.join(SP, 'plotly_data.json'), encoding='utf-8'))

#: '</script>' anywhere inside the JSON would close the tag early and silently
#: truncate the page. Escaping '<' is the standard guard; do it before embedding.
blob = json.dumps(data, separators=(',', ':')).replace('<', '\u003c')

for tag in ('</script', '</style'):
    assert tag not in blob, f"{tag} survived escaping"

html = (tpl
        + '\n<script>' + lib + '</script>\n'
        + '<script>window.__ATLAS__=' + blob + ';</script>\n'
        + '<script>' + charts + '</script>\n')

open(OUT, 'w', encoding='utf-8').write(html)

mb = len(html.encode('utf-8')) / 1e6
print(f"wrote {OUT}")
print(f"  {mb:.2f} MB total  (plotly {len(lib) / 1e6:.2f} MB, "
      f"data {len(blob) / 1e3:.1f} kB, page {len(tpl) / 1e3:.1f} kB)")
print(f"  plotly {plotly.__version__} from {JS}")
print(f"  {len(data['basins'])} basins, {len(data['named'])} named structures, "
      f"density {len(data['dens']['x'])}x{len(data['dens']['y'])}")
assert mb < 16, f"page is {mb:.1f} MB, over the 16 MB artifact cap"
