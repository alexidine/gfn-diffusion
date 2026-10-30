---
name: workvis
description: Refresh and show the owner's diagram of a live code workflow (unconditional or conditional crystal training, prior generation, crystal search), checked against the current code first. Use when the user invokes /workvis with a workflow name, e.g. "/workvis unconditional training", or asks to see, update or audit the diagram of one of these workflows. Not for one-off explanatory figures or wiki diagrams.
---

# workvis: workflow diagrams kept true to the code

The owner uses these diagrams to keep their mental model of a workflow current as
agents change the code. The diagram should be almost as good as reading the code,
and faster. Every claim on it is checked against the code before it is shown,
because a wrong box is worse than a missing one.

## Files: `energy_sampling/docs/workvis/`

| file | role |
|---|---|
| `<wf>.yaml` | the source. The only file you edit. |
| `render.py` | turns the source into `<wf>.html`, the viewer. Never edit the HTML by hand. |
| `scan.py` | compares the code under every anchor with `<wf>.stamp.json`. |
| `prev/<wf>.yaml` | the baseline that the highlights diff against. |
| `AUDITOR.md` | the brief for the independent auditor. |

Run both scripts with the bare `python` from `energy_sampling/`. They need only
`yaml` and `ast`.

## Workflows

| argument | `<wf>` | route |
|---|---|---|
| unconditional training | `unconditional` | committed `mk_dev.yaml`, protocol `unconditional_tb`: prior initialization, MLE warm start, equilibration |
| conditional training | `conditional` | not built yet. Same scope: prior initialization, MLE, equilibration, under the conditional protocol and flags |
| prior generation | `prior_generation` | not built yet |
| crystal search | `crystal_search` | not built yet |

A workflow pins the config values that define it (`route.pins`). Everything else
follows the committed config. If the committed config no longer carries a pinned
value, `scan.py` prints `ROUTE`. The diagram still describes the pinned route, so
resolve its values as the committed config with the pins applied.

## Refresh procedure

1. **Scan.** Run `python docs/workvis/scan.py <wf>` and read every line:
   - `CHANGED`: a function body moved.
   - `CALLS`: a call was added or dropped, the usual sign of a new branch.
   - `CONFIG`: a config value moved.
   - `MISSING`: an anchor no longer resolves.
   - `ROUTE`: the committed config no longer carries a pinned value.

   An unchanged scan proves only that the anchored code did not move. If the user
   asked for an audit, or behaviour may have moved through code the diagram does not
   anchor, continue to step 2 anyway.
2. **Nothing moved and nothing was asked:** do not touch the baseline. Re-render,
   open the viewer, and say the diagram is unchanged since its stamp.
3. **Baseline.** Before the first edit, copy `<wf>.yaml` to `prev/<wf>.yaml`. The
   highlights then show exactly this refresh's changes.
4. **Revise.** For each flagged anchor, read the change (`git diff <stamp
   commit> -- <file>` plus the working tree), then the code it reaches. Decide which
   claims are now wrong or missing, and edit the nodes and edges. Follow the content
   rules below.
5. **Audit.** Spawn ONE fresh general-purpose agent on Opus with the brief in
   `AUDITOR.md`. Pass it the changed node ids, or every node on a first build or
   when the user asks for a full audit. It must not be the agent that edited the
   diagram.
   - Fix every confirmed finding.
   - For a finding you doubt, read the code yourself and decide.
   - Then run `python docs/workvis/scan.py <wf> --stamp`, and set `stamp.commit`,
     `stamp.date` and `stamp.audited` in the YAML.
6. **Render and look.**
   - Run `python docs/workvis/render.py <wf>`.
   - Open `file:///.../docs/workvis/<wf>.html` with the browser pane's `preview_start`.
   - Confirm it did not print `render failed`, and take a screenshot.
7. **Report** in the AGENTS.md shape:
   - What changed in the diagram, meaning first. These are the changes to the
     owner's mental model.
   - What the auditor found and what you did about it.
   - What stays unverified. Values marked "derived" belong here.

## First build of a workflow

- Extract the content before drawing. Use one or two agents with narrow briefs,
  each returning the equations as TeX, resolved values with their config keys, and
  anchors with file:line.
- Spot-check the load-bearing claims yourself.
- Write the YAML, then audit every node.
- Show the owner one stage before building them all.

## Content rules (settled with the owner 2026-10-29)

- **Style.** Execution stages in the order they run. Stage groups hold boxes, and a
  per-step loop is drawn as its own group. Data stores are cylinders, the log Z
  parameter is a hexagon, and periodic controllers are teal. The node kinds are in
  `render.py::KINDS`.
- **Content.** Each box says what happens, as equations with their resolved
  hyperparameter values.
  - No function names in labels. They belong in `anchors`.
  - Stage and config names are fine where they name a thing the owner sets.
- **Scope.** Show only the path the requested route runs. Branches its config
  switches off are not drawn. Name a gate only where it explains why something
  shown does nothing.
- **Values.** Use the value the route resolves: stage block over base block over
  code default.
  - Label any number not read from code or config "derived, not measured".
  - Mark any claim still unconfirmed in its line.
- **Math.**
  - Write TeX between single dollars, in single-quoted YAML strings (`''` is a
    literal quote).
  - Never put a double quote inside math.
  - Never put a `|` in an edge label.
  - Write a batch mean as $\langle\cdot\rangle$.
- **Anchors.** Every node anchors each function and config key its claims rest on.
  This is what lets `scan.py` notice when the box goes stale.
- **Density.** The owner zooms in deep and pans with the ThinkPad TrackPoint's
  middle-button scroll, which reaches the page as wheel events. The viewer therefore
  pans on a plain wheel and zooms on Ctrl+wheel; the panel has a toggle for a mouse.
  A minimap and scrollbars give landmarks. Dense boxes are fine, but never truncate a
  claim to save space.
- **Titles carry the overview.** Zoomed out, the viewer shows each box's title alone
  (semantic zoom, the `semanticZoom` script in `render.py`), so a title must say what the box does
  on its own. Short titles stay legible at lower zoom.
- **Highlights.** Changed lines show bold and coloured against `prev/<wf>.yaml`.
  `render.py` does this, so never mark changes by hand.

## Status

The diagrams are derived views of current behaviour: observations at a stamped
commit. The code stays the authority. If the diagram and a wiki page disagree,
check the code, and report the disagreement as AGENTS.md asks.
