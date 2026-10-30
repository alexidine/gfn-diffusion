"""Load a workflow diagram's source, merging a variant onto the diagram it extends.

A variant (`unconditional_nf` of `unconditional`) names its base with `extends:` and
carries its own header -- workflow, title, subtitle, route, stamp -- plus only what
differs from the base:

  nodes         a node whose id the base has replaces that node in place; a new id is
                added after the node its `after:` key names (else at the end). A node
                with a `patch:` edits the base node instead of replacing it:
                  replace      [[text, new line], ...]: the one base line containing text
                  add          lines appended
                  drop         [text, ...]: the one base line containing each text
                  title, kind  replace the base's
                  anchors_add, anchors_drop
                so the base's other lines stay live. A text that matches no base line, or
                more than one, raises: the base line it edited has changed, and the
                variant's edit needs re-reading. A patched line that restates a base
                anchor's claim for this route lists that anchor in anchors_add too: it
                is then the variant's to stamp, and scan.py stops reporting it as an
                inherited claim that reads a changed value.
  remove_nodes  ids of base nodes the variant's route does not run; their edges go too
  groups        a group whose id the base has replaces it; a new id is added
  edges         an edge whose (from, to) the base has replaces it in place; others are added
  remove_edges  [{from, to}] base edges to drop
  triage        what `scan.py --triage` reads: `swaps` (the base's component and the
                variant's, as symbol anchors) and `terms` (strings whose presence in a
                box's lines marks it as worth re-reading)

So a box the two routes share lives in one file: it is edited, scanned, audited and
stamped once, in the base. render.py draws the merged diagram; scan.py stamps a
variant's own anchors and checks its inherited boxes against its route.

Resolution is yaml only.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
HEADER = ('workflow', 'title', 'subtitle', 'route', 'stamp')


def read(name: str) -> dict:
    """The source file of one workflow, unmerged."""
    with open(HERE / f'{name}.yaml', encoding='utf-8') as fh:
        return yaml.safe_load(fh)


def load(name: str, _chain: tuple = ()) -> dict:
    """The diagram as drawn: a variant merged onto its base, recursively."""
    if name in _chain:
        raise ValueError(f'extends cycle: {" -> ".join(_chain + (name,))}')
    src = read(name)
    if not src.get('extends'):
        return src
    return merge(load(src['extends'], _chain + (name,)), src)


def own_node_ids(name: str) -> list[str]:
    """Ids of the nodes this workflow's own file defines (all of them, for a base)."""
    return [n['id'] for n in read(name).get('nodes') or []]


def _by_id(items: list[dict], what: str) -> dict:
    out = {}
    for it in items:
        if it['id'] in out:
            raise ValueError(f'duplicate {what} id {it["id"]!r}')
        out[it['id']] = it
    return out


def _one_line(lines: list[str], text: str, nid: str) -> int:
    hits = [i for i, line in enumerate(lines) if text in line]
    if len(hits) != 1:
        raise ValueError(f'node {nid!r} patch: {len(hits)} base lines contain {text!r} (need exactly 1)')
    return hits[0]


def apply_patch(node: dict, patch: dict) -> dict:
    """The base node with a variant's line-level edits applied."""
    out = copy.deepcopy(node)
    nid, lines = out['id'], list(out.get('lines') or [])
    unknown = set(patch) - {'replace', 'add', 'drop', 'title', 'kind', 'anchors_add', 'anchors_drop'}
    if unknown:
        raise ValueError(f'node {nid!r} patch: unknown keys {sorted(unknown)}')
    for text, new in patch.get('replace') or []:
        lines[_one_line(lines, text, nid)] = new
    for text in patch.get('drop') or []:
        del lines[_one_line(lines, text, nid)]
    lines.extend(patch.get('add') or [])
    out['lines'] = lines
    for key in ('title', 'kind'):
        if key in patch:
            out[key] = patch[key]
    anchors = [a for a in out.get('anchors') or [] if a not in set(patch.get('anchors_drop') or [])]
    missing = set(patch.get('anchors_drop') or []) - set(out.get('anchors') or [])
    if missing:
        raise ValueError(f'node {nid!r} patch: anchors_drop names anchors the base node lacks: {sorted(missing)}')
    out['anchors'] = anchors + [a for a in patch.get('anchors_add') or [] if a not in anchors]
    return out


def own_anchors(name: str) -> dict[str, list[str]]:
    """{node id: the anchors this workflow's own file adds}: every anchor of a node it defines or
    replaces, the anchors_add of a node it patches. A base node's own anchors are its own."""
    out = {}
    for n in read(name).get('nodes') or []:
        out[n['id']] = list((n['patch'].get('anchors_add') or []) if 'patch' in n else (n.get('anchors') or []))
    return out


def merge(base: dict, var: dict) -> dict:
    out = copy.deepcopy({k: v for k, v in base.items() if k not in ('nodes', 'edges', 'groups', 'triage')})
    out.update(copy.deepcopy({k: var[k] for k in HEADER if k in var}))
    out['extends'] = var['extends']
    if 'triage' in var:
        out['triage'] = copy.deepcopy(var['triage'])

    groups = copy.deepcopy(base.get('groups') or [])
    for g in var.get('groups') or []:
        ids = [x['id'] for x in groups]
        if g['id'] in ids:
            groups[ids.index(g['id'])] = copy.deepcopy(g)
        else:
            groups.append(copy.deepcopy(g))
    out['groups'] = groups

    nodes = copy.deepcopy(base.get('nodes') or [])
    _by_id(nodes, 'base node')
    for n in var.get('nodes') or []:
        n = copy.deepcopy(n)
        after = n.pop('after', None)
        ids = [x['id'] for x in nodes]
        if 'patch' in n:
            if n['id'] not in ids or after is not None or set(n) - {'id', 'patch'}:
                raise ValueError(f'node {n["id"]!r}: a patch edits a base node in place and carries only id and patch')
            nodes[ids.index(n['id'])] = apply_patch(nodes[ids.index(n['id'])], n['patch'])
        elif n['id'] in ids:
            if after is not None:
                raise ValueError(f'node {n["id"]!r} replaces a base node in place: drop its after:')
            nodes[ids.index(n['id'])] = n
        elif after is not None:
            if after not in ids:
                raise ValueError(f'node {n["id"]!r}: after {after!r} names no node')
            nodes.insert(ids.index(after) + 1, n)
        else:
            nodes.append(n)
    removed = set(var.get('remove_nodes') or [])
    missing = removed - {x['id'] for x in nodes}
    if missing:
        raise ValueError(f'remove_nodes names no node: {sorted(missing)}')
    out['nodes'] = [n for n in nodes if n['id'] not in removed]
    live = {n['id'] for n in out['nodes']}

    edges = copy.deepcopy(base.get('edges') or [])
    keys = [(e['from'], e['to']) for e in edges]
    for e in var.get('edges') or []:
        k = (e['from'], e['to'])
        if keys.count(k) > 1:
            raise ValueError(f'edge {k}: the base has it more than once, so a replacement is ambiguous')
        if k in keys:
            edges[keys.index(k)] = copy.deepcopy(e)
        else:
            edges.append(copy.deepcopy(e))
            keys.append(k)
    drop = {(e['from'], e['to']) for e in var.get('remove_edges') or []}
    missing = drop - set(keys)
    if missing:
        raise ValueError(f'remove_edges names no edge: {sorted(missing)}')
    out['edges'] = [e for e in edges if (e['from'], e['to']) not in drop
                    and e['from'] in live and e['to'] in live]
    return out
