# Workflow diagram auditor brief

The prompt `/workvis` hands to ONE fresh agent after it revises a diagram. The agent
did not draw the diagram, and must not take it on trust: the owner builds their
mental model of the code from it, so a wrong box is worse than a missing one.
Fill the three placeholders and pass everything below the line as the prompt.

---

Audit a workflow diagram against the code. Read-only: do not edit, create or delete
any file. Light CPU only: never run train.py, nothing on GPU.

Work from `C:\Users\mikem\Projects\mxt_gfn\gfn_diffusion\energy_sampling`; the
repository instructions are `../AGENTS.md`. The diagram source is
`docs/workvis/{WORKFLOW}.yaml`: each node has a `title`, `lines` (claims, math in TeX
between single dollars) and `anchors` (the functions, in `path::Class.method`
notation, and config keys, as `cfg:key` or `cfg:@stage.key` for a key of the named
stage, that its claims rest on). Edges claim order (`flow`), data movement (`data`)
and writes (`write`). The route is the YAML's `route`: its config, with any
`route.overrides` set on top, and the pinned values; every value the diagram states
must be the one THIS route resolves (stage block over base block over code default).
Resolve values with
`CUDA_VISIBLE_DEVICES=-1 C:/Users/mikem/venvs/csd_mxt_gfn/Scripts/python.exe -m config_snapshot configs/mk_dev.yaml`
with PYTHONPATH including `../../mxtaltools` (the bare `python` has no torch); for a
route with overrides, run it on the copy with the overrides applied that the context
below names. The
code in the working tree is the authority; wiki pages under `docs/wiki/` are leads only.

Audit these nodes: {NODES}
Context for this audit: {CONTEXT}

Report three sections, under ~900 words in total.

1. **Claims.** For every audited node, check every claim: each equation (symbols,
   signs, constants, what is detached), each value, the order its edges assert, and
   each data edge. List only the claims that are WRONG or UNVERIFIABLE, as
   `node id: claim -> what the code does (file:line)`. Then one line giving the
   count of claims checked and found right. Do not restate correct claims.
2. **Coverage.** Starting from the route's entry points, name up to five behaviours
   that execute on this route, would change a reader's model of what happens, and
   that the diagram does not show. Each one gets a line, a file:line, and the node it
   belongs beside. Do not list branches the route's config switches off.
3. **Anchors.** For each audited node, name the functions or config keys its claims
   rest on that are not in its `anchors`, and any anchor that its claims do not use.

Mark anything you could not settle as UNCONFIRMED. Do not speculate, and do not
propose wording beyond the correction itself.
