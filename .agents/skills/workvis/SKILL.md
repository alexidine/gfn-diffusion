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
| `diagram.py` | loads a source, merging a variant onto the diagram it `extends`. Its docstring is the variant syntax. |
| `render.py` | turns the source into `<wf>.html`, the viewer. Never edit the HTML by hand. |
| `scan.py` | compares the code under the anchors with `<wf>.stamp.json`; `--diff` and `--triage` below. |
| `prev/<wf>.yaml` | a base diagram's highlight baseline. A variant has none: it highlights against its base. |
| `AUDITOR.md` | the brief for the independent auditor. |

Run the scripts with the bare `python` from `energy_sampling/`. They need only
`yaml` and `ast`.

Cost discipline: a box lives in one file, a refresh reads only the function bodies
that moved, and an agent is briefed with only the boxes a static pass flags.
Spend agents on the first build of a base diagram and on owner-requested full
audits, not on re-deriving what did not change.

## Workflows

| argument | `<wf>` | route |
|---|---|---|
| unconditional training | `unconditional` | committed `mk_dev.yaml`, protocol `unconditional_tb`: prior initialization, MLE warm start, equilibration |
| unconditional NF / NF AR training | `unconditional_nf` | variant of `unconditional`: overrides `model.kind: nf` and `gradient_norm_clip: 10.0` (the committed `auto` is refused under nf) |
| conditional training | `conditional` | not built yet. Same scope, under the conditional protocol and flags; build it as a variant of `unconditional` if most boxes carry over |
| prior generation | `prior_generation` | not built yet |
| crystal search | `crystal_search` | not built yet |

A workflow pins the config values that define it (`route.pins`), values the
committed config must itself carry. A route that deliberately runs with other
values lists them as `route.overrides`; `scan.py` resolves config anchors with them
applied, and the viewer's panel shows them. Everything else follows the committed
config. `scan.py` prints `ROUTE` when the committed config no longer carries a pin,
or now carries an override (which can then become a pin). For an override route,
auditors need the config as a file: write the committed config with the overrides
set to the session scratchpad and name it in the audit context.

A **variant** (`extends: <base>`) holds only its own boxes: those it replaces, those
it patches line by line (`patch:`), new ones (`after:`), and its edge changes. A box
the two routes share lives in the base and is edited, audited and stamped there
once. A variant stamps only the anchors its own file adds. Prefer a patch when most
of a base box's lines still hold, so the base's other lines stay live; a patch whose
target line has changed in the base raises, which is the signal to re-read it.

## Refresh procedure

1. **Scan.** Run `python docs/workvis/scan.py <wf> --diff` and read every line:
   - `CHANGED`: a function body moved; the indented diff below it is that body's
     change since the stamped commit. Read it instead of a whole-file `git diff`.
   - `CALLS`: a call was added or dropped, the usual sign of a new branch.
   - `CONFIG`: a config value moved.
   - `MISSING`: an anchor no longer resolves.
   - `ROUTE`: the committed config no longer carries a pin, or carries an override;
     for a variant, also an inherited box that reads a value this route changes:
     override or patch that box.
   - `BASE` (variants): the base has drifted. Refresh the base first, then re-read
     the variant's own versions of the boxes it names.

   An unchanged scan proves only that the anchored code did not move. If the user
   asked for an audit, or behaviour may have moved through code the diagram does not
   anchor, continue to step 2 anyway.
2. **Nothing moved and nothing was asked:** do not touch the baseline. Re-render,
   run the render check (step 6), and say the diagram is unchanged since its stamp.
3. **Baseline (base diagrams).** Before the first edit, copy `<wf>.yaml` to
   `prev/<wf>.yaml`. The highlights then show exactly this refresh's changes.
4. **Revise.** For each flagged anchor, start from its `--diff` hunk; read further
   code only where the hunk reaches it. Decide which claims are now wrong or
   missing, and edit the nodes and edges. Follow the content rules below.
5. **Audit.** Spawn ONE fresh general-purpose agent on Opus with the brief in
   `AUDITOR.md`. Pass it the changed node ids only, or every node on a base's first
   build or when the user asks for a full audit. For a variant, its own boxes are
   the scope; inherited boxes are audited in the base. It must not be the agent
   that edited the diagram.
   - Fix every confirmed finding. A finding in an inherited box goes to the base.
   - For a finding you doubt, read the code yourself and decide.
   - Then run `python docs/workvis/scan.py <wf> --stamp` (it refuses while `ROUTE`
     names an inherited box), and set `stamp.commit`, `stamp.date` and
     `stamp.audited` in the YAML.
6. **Render and check.**
   - Run `python docs/workvis/render.py <wf>`.
   - Open `file:///.../docs/workvis/<wf>.html` with the browser pane's
     `preview_start`, then one `javascript_tool` call:
     `await new Promise(r => { const t = setInterval(() => { if (window.WORKVIS_CHECK) { clearInterval(t); r(); } }, 100); }); window.WORKVIS_CHECK`
     It must show `ok: true` (every box drawn, no math errors).
   - Take a screenshot only when `render.py` itself changed.
7. **Report** in the AGENTS.md shape:
   - What changed in the diagram, meaning first. These are the changes to the
     owner's mental model.
   - What the auditor found and what you did about it.
   - What stays unverified. Values marked "derived" belong here.

## First build of a base workflow

- Extract the content before drawing. Use one or two agents with narrow briefs,
  each returning the equations as TeX, resolved values with their config keys, and
  anchors with file:line.
- Spot-check the load-bearing claims yourself.
- Write the YAML, then audit every node.
- Show the owner one stage before building them all.

## First build of a variant

1. Write the variant's header with no nodes: `workflow`, `extends`, `title`,
   `subtitle`, `route` (pins and overrides), `stamp`, and `triage`: `swaps` (the
   base's component and the variant's, as symbol anchors, e.g. the two policy
   classes) and `terms` (strings in a box's lines that mark it as depending on the
   swapped part).
2. Check the route config: write it to the scratchpad with the overrides applied
   and run `config_snapshot --check` on it.
3. Run `python docs/workvis/scan.py <wf> --triage`. It lists FLAGGED boxes (a changed
   config value, an anchor in or a call within two levels into the swapped code, or
   a term) and WEAK ones (a call three levels down). On the NF variant it flagged 18
   of 37 boxes, 15 of the 16 that needed changes; the 16th was WEAK.
4. ONE extraction agent, briefed with that list: FLAGGED boxes checked in full, WEAK
   ones briefly, unflagged ones not re-derived. It answers per box SAME, or the new
   lines with values and anchors, or NEW boxes with their edges.
5. Write the variant's boxes (patch where most base lines hold), run
   `scan.py <wf>`, which must print no `ROUTE` line, then audit the variant's own
   boxes with the triage list as context, and render with the check above.

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
  This is what lets `scan.py` notice when the box goes stale. A patched line that
  restates a base anchor's claim for the variant's route lists that anchor in the
  patch's `anchors_add`.
- **Density.** The owner zooms in deep and pans with the ThinkPad TrackPoint's
  middle-button scroll, which reaches the page as wheel events. The viewer therefore
  pans on a plain wheel and zooms on Ctrl+wheel; the panel has a toggle for a mouse.
  A minimap and scrollbars give landmarks. Dense boxes are fine, but never truncate a
  claim to save space.
- **Titles carry the overview.** Zoomed out, the viewer shows each box's title alone
  (semantic zoom, the `semanticZoom` script in `render.py`), so a title must say what
  the box does on its own. Short titles stay legible at lower zoom.
- **Highlights.** A base diagram's changed lines show bold and coloured against
  `prev/<wf>.yaml`; a variant's, against its base. `render.py` does this, so never
  mark changes by hand.

## Status

The diagrams are derived views of current behaviour: observations at a stamped
commit. The code stays the authority. If the diagram and a wiki page disagree,
check the code, and report the disagreement as AGENTS.md asks.
