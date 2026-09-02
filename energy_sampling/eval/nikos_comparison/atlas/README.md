# atlas — the acridine sg14-Z'1 landscape document

Produces **acridine_packing_atlas.html**, published 2026-08-26 at
<https://claude.ai/code/artifact/da8c5f7a-e973-488b-9962-821d3ff22770>.

These scripts were written in a session scratchpad and moved here so the document is
reproducible. Each one sets `SP = os.path.dirname(os.path.abspath(__file__))`, so
every intermediate lands **in this directory** — artifacts beside their producer.
The intermediates are NOT committed; they all regenerate from the cached tensors.

## Inputs (not in the repo)

    D:\crystal_datasets\acridine\nikos_comparison\lowE_dmat.pt
        1,969 x 1,969 envwise RDF distances over the mace <= -60 stratum, plus
        energies. Every basin in this document comes from this one matrix.
    D:\crystal_datasets\acridine\nikos_comparison\nikos_levels.pt
        Nikos' structures at L0 / L1 / L2.
    D:\crystal_datasets\acridine\nikos_comparison\lowE_descriptors.pt
    D:\crystal_datasets\acridine\prior_chunks\may_acridine_sg14_zp1_*.pt
        The raw search output, 55,076 samples -> 53,582 physical.

## Run order

    python relax_states.py        -> relax_states.json      rigid-body relax of the
                                                            named structures, L0->L2
    python mds_map.py             -> lowE_mds.json          classical MDS of the RDF
                                                            distance matrix
    python dump_shells.py         -> shells.json            coordination shells for
                                                            the controls / Nikos set
    python projection_quality.py  -> projection.json        per-structure density and
                                                            energy for panel A
    python coarse_landscape.py    -> coarse_landscape.json  THE BASIN DEFINITION:
                                                            complete linkage, cut 0.10
    python tight_regions.py       -> tight_regions.json     are the clumps real?

    python build_figure.py        -> figure_block.html
    python build_coarse_fig.py    -> coarse_block.html
    python build_clumps_block.py  -> clumps_block.html
    python build_mds_panels.py    -> mds_block.html
    python build_atlas.py         -> acridine_packing_atlas.html

The Plotly version of the same figures -- plain interactive charts instead of
hand-drawn SVG, which is what buys hover identity on 42 points:

    python export_plotly_data.py  -> plotly_data.json   (bins the 53,582-structure
                                                         landscape server-side)
    python build_plotly_page.py   -> acridine_basin_figures.html

Static PNGs of those same five figures, via kaleido:

    python export_figures.py           -> figures/*.png        (light, scale 2)
    python export_figures.py --dark    -> figures/*_dark.png

Two ramps, both documented steps of the one blue scale: the density heatmap uses the
FULL range so an empty cell recedes into the surface, while the discrete basin
markers obey the ordinal floor (light: no lighter than step 250; dark: no darker
than 600) -- at the full range the shallowest basins were nearly invisible on white.

`build_plotly_page.py` INLINES plotly.min.js from the installed plotly package
(~4.8 MB page total) because an artifact's CSP blocks CDN scripts. It reads
`page_template.html` and `page_charts.js`. Note the two version numbers differ and
both matter: plotly.py 6.3.0 ships plotly.js 3.1.0.

**Keep `page_charts.js` pure ASCII.** The page is wrapped by the artifact host and
cannot guarantee a `<meta charset>` inside the first 1024 bytes, so a literal Greek
letter renders as mojibake under a windows-1252 default. Use \uXXXX escapes; the
builder does not convert for you.

`python` on PATH has no torch, and **PYTHONPATH is required** -- everything here
reaches `mxtaltools`, which is a sibling repo and is not installed into the venv.
Without it you get `ModuleNotFoundError: No module named 'mxtaltools'` from the
producers and from `build_mds_panels.py`; the three pure-JSON renderers happen to
run anyway, which makes the omission look like a partial success.

```powershell
$env:PYTHONPATH = "C:\Users\mikem\Projects\mxt_gfn\mxtaltools;C:\Users\mikem\Projects\mxt_gfn\gfn_diffusion"
cd C:\Users\mikem\Projects\mxt_gfn\gfn_diffusion
& "C:\Users\mikem\venvs\csd_mxt_gfn\Scripts\python.exe" energy_sampling\eval\nikos_comparison\atlas\coarse_landscape.py
```

Verified 2026-08-26: all 11 scripts import and the five renderers reproduce the
published document byte-for-byte (717,509 bytes) from this directory.

## The one thing to know before changing anything

**`coarse_landscape.json` is the single basin definition** — complete linkage on the
RDF distances at the calibrated 0.10 cut, 42 basins. Everything that draws a basin
reads it: the plate grid, panel B, the four coarse views. It was not always so; an
earlier pass had the plate grid on a 34-basin *average*-linkage clustering while the
coarse figure used the 42, and the document quoted both counts.

Use complete linkage. It BOUNDS each basin's diameter by the cut (widest 0.0937,
inside the certain-match band), which is what the pairwise calibration licenses.
Average linkage chains: it produced a 1,167-member group whose mean internal distance
(0.047) exceeded its distance to the nearest other group (0.020) — not a basin.
Complete's price is over-splitting, and it is visible here: basins 18 and 16 sit
0.0082 apart, one basin cut in two. The count is soft between ~30 and 45.

**Basin IDs are not portable across linkage.** Any basin number in a note written
before 2026-08-26 refers to the average-linkage labelling.

## Not in the closure

The scratchpad also held ~55 one-off probes (`probe_*.py`, `diag_*.py`,
`zp2_sanity.py`, ...). They were exploratory, are not needed to rebuild the document,
and were deliberately left behind.
