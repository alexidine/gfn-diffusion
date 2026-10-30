"""Change scanner for the workflow diagrams: what moved under a diagram since it was last audited.

    python docs/workvis/scan.py unconditional            # report against unconditional.stamp.json
    python docs/workvis/scan.py unconditional --diff     # ... with each changed function's body diff
    python docs/workvis/scan.py unconditional --stamp    # record the current state (after an audit)
    python docs/workvis/scan.py unconditional_nf --triage   # a variant: which inherited boxes to re-read

Every node of <workflow>.yaml may carry `anchors`, in the wiki's notation:

  symbol  `train.py::Modeller.fused_train_step`, `models/gfn.py::GFN.get_traj_fwd`; a path
          beginning `mxtaltools/` resolves in the sibling repository.
  config  `cfg:a.b.c`, a key of the route's config; `cfg:@stage.key`, a key of the named
          stage of the protocol the route pins.

The route is the committed config with the YAML's `route.overrides` set on top (values
the route deliberately runs with, e.g. another model kind); `route.pins` are values the
committed config itself must carry. Config anchors resolve on the overridden config.

The stamp records, per symbol, a hash of its parsed body (docstring dropped, so comments,
docstrings and formatting never trip it) and the calls it makes; per config key, its
value; and the commit of each repository. A scan reports, by node: CHANGED (body hash
moved; --diff prints the body's diff since the stamped commit), CALLS (a call added or
dropped -- the usual sign of a new branch the diagram may not show), CONFIG (value
moved), MISSING (anchor no longer resolves), and ROUTE (the committed config no longer
carries a value the route pins, or now carries an override, which can become a pin).
Exit 0 clean, 1 drift, 2 missing anchors or no stamp.

A VARIANT (`extends:`, see diagram.py) stamps only the anchors its own file adds; the
boxes it inherits are its base's, stamped there. Its scan also reports the base's drift
(BASE), and an inherited box that reads a config value this route changes (ROUTE: the
variant must override that box). --triage lists, for each inherited box, why it may not
hold on this route: a config value it reads differs, it anchors or calls into the
swapped component (`triage.swaps`), or its lines mention one of `triage.terms`. Triage
is a static reading aid: an unflagged box can still differ.

The scan is a prompt to re-read code, not a verdict: an unchanged hash proves only
that the anchored functions did not change. Behaviour can move through code the
diagram does not anchor; the auditor's coverage pass is the check for that.

Resolution is `ast` and `yaml` only: nothing here imports torch or the trainer.
"""

from __future__ import annotations

import argparse
import ast
import copy
import datetime as dt
import difflib
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ENERGY_ROOT = HERE.parents[1]                      # .../gfn_diffusion/energy_sampling
MXT_ROOT = ENERGY_ROOT.parents[1] / 'mxtaltools'   # .../mxt_gfn/mxtaltools
sys.path.insert(0, str(HERE))
import diagram  # noqa: E402

DIFF_LINES = 60   # --diff prints at most this many lines per changed body

_TREES: dict[Path, ast.Module] = {}


def _tree(path: Path) -> ast.Module:
    if path not in _TREES:
        _TREES[path] = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    return _TREES[path]


def _file_for(rel: str) -> Path:
    if rel.startswith('mxtaltools/'):
        return MXT_ROOT / rel
    return ENERGY_ROOT / rel


def _find(body: list[ast.stmt], name: str) -> ast.AST | None:
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
            return node
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id == name for t in targets):
                return node
    return None


def _walk(tree: ast.AST, qual: str, rel: str) -> ast.AST:
    node = tree
    for part in qual.split('.'):
        found = _find(getattr(node, 'body', []), part)
        if found is None:
            raise LookupError(f'{qual}: no {part!r} in {rel}')
        node = found
    return node


def resolve_symbol(anchor: str) -> ast.AST:
    rel, _, qual = anchor.partition('::')
    path = _file_for(rel)
    if not path.exists():
        raise LookupError(f'no file {rel}')
    return _walk(_tree(path), qual, rel)


def _calls(node: ast.AST) -> list[str]:
    """Dotted names of the calls a definition makes: `self.x.y(...)` and bare `f(...)`."""
    out = set()
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        parts, f = [], sub.func
        while isinstance(f, ast.Attribute):
            parts.append(f.attr)
            f = f.value
        if isinstance(f, ast.Name) and (f.id == 'self' or not parts):
            out.add('.'.join([f.id] + parts[::-1]))
    return sorted(out)


def _digest(node: ast.AST) -> str:
    node = copy.deepcopy(node)
    body = getattr(node, 'body', None)
    if (isinstance(body, list) and body and isinstance(body[0], ast.Expr)
            and isinstance(getattr(body[0], 'value', None), ast.Constant)
            and isinstance(body[0].value.value, str)):
        node.body = body[1:] or [ast.Pass()]
    return hashlib.sha1(ast.dump(node, include_attributes=False).encode()).hexdigest()[:12]


def fingerprint(anchor: str) -> dict:
    node = resolve_symbol(anchor)
    return {'hash': _digest(node), 'calls': _calls(node)}


# ---------------------------------------------------------------- config

def _dotted(cfg: dict, key: str):
    node = cfg
    for part in key.split('.'):
        if not isinstance(node, dict) or part not in node:
            raise LookupError(f'no config key {key}')
        node = node[part]
    return node


def resolve_config(anchor: str, cfg: dict, protocol: str | None):
    key = anchor[len('cfg:'):]
    if key.startswith('@'):
        stage_name, _, rest = key[1:].partition('.')
        stages = _dotted(cfg, f'protocols.{protocol}.stages')
        stage = next((s for s in stages if s.get('name') == stage_name), None)
        if stage is None:
            raise LookupError(f'protocol {protocol} has no stage {stage_name!r}')
        return _dotted(stage, rest) if rest else stage
    return _dotted(cfg, key)


def route_config(route: dict) -> tuple[dict, dict, str | None]:
    """The committed config, the route's config (the committed one with route.overrides set), its protocol."""
    committed = yaml.safe_load((ENERGY_ROOT / route.get('config', 'configs/mk_dev.yaml')).read_text(encoding='utf-8'))
    cfg = copy.deepcopy(committed)
    for key, value in (route.get('overrides') or {}).items():
        *parents, leaf = key.split('.')
        node = cfg
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = value
    protocol = (route.get('pins') or {}).get('protocol') or cfg.get('protocol')
    return committed, cfg, protocol


def route_mismatches(route: dict) -> list[str]:
    cfg, _, _ = route_config(route)
    out = []
    for key, want in (route.get('pins') or {}).items():
        try:
            have = _dotted(cfg, key)
        except LookupError:
            have = '<absent>'
        if have != want:
            out.append(f'{key}: committed config has {have!r}, the route pins {want!r}')
    for key, want in (route.get('overrides') or {}).items():
        try:
            if _dotted(cfg, key) == want:
                out.append(f'{key}: committed config now carries the override {want!r}; make it a pin')
        except LookupError:
            pass
    return out


def _value(anchor: str, route: dict):
    _, cfg, protocol = route_config(route)
    try:
        return resolve_config(anchor, cfg, protocol)
    except LookupError:
        return '<absent>'


# ---------------------------------------------------------------- state

def _git(*args: str, cwd: Path = ENERGY_ROOT) -> str | None:
    try:
        return subprocess.run(['git', *args], cwd=cwd, capture_output=True, text=True,
                              encoding='utf-8', check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def _head(cwd: Path) -> str:
    return (_git('rev-parse', '--short', 'HEAD', cwd=cwd) or '').strip()


def current_state(name: str) -> tuple[dict, list[tuple[str, str, str]]]:
    """{anchor: fingerprint or {'value': ...}} over the anchors this workflow's own file adds,
    and the anchors that failed to resolve."""
    route = diagram.read(name).get('route', {})
    _, cfg, protocol = route_config(route)
    state, missing = {}, []
    for node_id, anchors in diagram.own_anchors(name).items():
        for anchor in anchors:
            if anchor in state:
                continue
            try:
                if anchor.startswith('cfg:'):
                    state[anchor] = {'value': resolve_config(anchor, cfg, protocol)}
                else:
                    state[anchor] = fingerprint(anchor)
            except (LookupError, SyntaxError) as e:
                missing.append((node_id, anchor, str(e)))
    return state, missing


def inherited_conflicts(name: str) -> list[tuple[str, str, object, object]]:
    """A variant's inherited claims that read a config value its route changes:
    (node id, anchor, base route's value, this route's value)."""
    src = diagram.read(name)
    if not src.get('extends'):
        return []
    base_route = diagram.load(src['extends'])['route']
    own = diagram.own_anchors(name)
    replaced = {n['id'] for n in src.get('nodes') or [] if 'patch' not in n}
    out = []
    for node in diagram.load(name)['nodes']:
        if node['id'] in replaced:
            continue
        for anchor in node.get('anchors') or []:
            if anchor.startswith('cfg:') and anchor not in own.get(node['id'], []):
                here, there = _value(anchor, src['route']), _value(anchor, base_route)
                if here != there:
                    out.append((node['id'], anchor, there, here))
    return out


# ---------------------------------------------------------------- diffs

def body_diff(anchor: str, stamp: dict, then_hash: str | None) -> list[str]:
    """The anchored body's diff from the stamped commit to the working tree."""
    rel, _, qual = anchor.partition('::')
    mxt = rel.startswith('mxtaltools/')
    commit = stamp.get('mxt_commit' if mxt else 'commit')
    if not commit:
        return ['    (the stamp records no commit for this repository: re-stamp to enable --diff)']
    old_src = _git('show', f'{commit}:./{rel}', cwd=MXT_ROOT if mxt else ENERGY_ROOT)
    if old_src is None:
        return [f'    ({rel} is not in {commit})']
    try:
        old_node = _walk(ast.parse(old_src), qual, rel)
    except LookupError:
        return [f'    ({qual} is not in {rel} at {commit})']
    new_node = resolve_symbol(anchor)
    old_seg = ast.get_source_segment(old_src, old_node) or ''
    new_seg = ast.get_source_segment(_file_for(rel).read_text(encoding='utf-8'), new_node) or ''
    out = []
    if then_hash and _digest(old_node) != then_hash:
        out.append(f'    (the stamped body was a working-tree state; this diff is from {commit})')
    lines = list(difflib.unified_diff(old_seg.splitlines(), new_seg.splitlines(), lineterm='', n=1))[2:]
    out += ['    ' + line for line in lines[:DIFF_LINES]]
    if len(lines) > DIFF_LINES:
        out.append(f'    ... {len(lines) - DIFF_LINES} more lines: git diff {commit} -- {rel}')
    return out


def drift_report(name: str, show_diff: bool = False) -> tuple[list[str], int, list[str], int]:
    """(lines, drifted anchor count, ids of the nodes they belong to, missing count)."""
    stamp_path = HERE / f'{name}.stamp.json'
    state, missing = current_state(name)
    lines = [f'MISSING  {node_id:<12} {anchor}  ({why})' for node_id, anchor, why in missing]
    if not stamp_path.exists():
        return lines + [f'no stamp for {name}: audit the diagram, then run with --stamp'], 0, [], len(missing) or 1
    stamp = json.loads(stamp_path.read_text(encoding='utf-8'))
    owners: dict[str, list[str]] = {}
    for node_id, anchors in diagram.own_anchors(name).items():
        for anchor in anchors:
            owners.setdefault(anchor, []).append(node_id)
    drift, nodes = 0, set()
    for anchor, now in state.items():
        then = stamp['anchors'].get(anchor)
        who = ','.join(owners.get(anchor, []))
        if then is None:
            lines.append(f'NEW      {who:<12} {anchor}  (not in the stamp)')
        elif 'value' in now:
            if now['value'] == then.get('value'):
                continue
            lines.append(f'CONFIG   {who:<12} {anchor}  {then.get("value")!r} -> {now["value"]!r}')
        elif now['hash'] != then.get('hash'):
            added = sorted(set(now['calls']) - set(then.get('calls', [])))
            dropped = sorted(set(then.get('calls', [])) - set(now['calls']))
            lines.append(f'CHANGED  {who:<12} {anchor}')
            if added or dropped:
                lines.append(f'  CALLS  +{", ".join(added) or "none"}  -{", ".join(dropped) or "none"}')
            if show_diff:
                lines += body_diff(anchor, stamp, then.get('hash'))
        else:
            continue
        drift += 1
        nodes.update(owners.get(anchor, []))
    return lines, drift, sorted(nodes), len(missing)


# ---------------------------------------------------------------- triage

_STOP = {'forward', 'to', 'train', 'eval', 'parameters', 'named_parameters', 'state_dict', 'load_state_dict',
         'modules', 'children', 'cuda', 'cpu', 'apply', 'zero_grad', 'register_buffer'}


def _methods(anchor: str) -> set[str]:
    node = resolve_symbol(anchor)
    return {f.name for f in getattr(node, 'body', []) if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))
            and not f.name.startswith('__')} - _STOP


def _reached(anchor: str, depth: int) -> set[str]:
    """Method names the anchored function calls on any receiver but bare self, following its own
    self.m(...) calls into sibling methods of the same class `depth` levels deep."""
    rel, _, qual = anchor.partition('::')
    try:
        node = resolve_symbol(anchor)
    except (LookupError, SyntaxError):
        return set()
    out, own = set(), set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
            recv = sub.func.value
            if isinstance(recv, ast.Name) and recv.id == 'self':
                own.add(sub.func.attr)
            else:
                out.add(sub.func.attr)
    if depth > 1 and '.' in qual:
        cls = qual.rsplit('.', 1)[0]
        for m in own:
            out |= _reached(f'{rel}::{cls}.{m}', depth - 1)
    return out


def triage(name: str) -> list[str]:
    src = diagram.read(name)
    if not src.get('extends'):
        return [f'{name} extends nothing: triage reads a variant against its base']
    spec = src.get('triage') or {}
    swaps, terms = spec.get('swaps') or [], spec.get('terms') or []
    methods = set().union(*(_methods(a) for a in swaps)) if swaps else set()
    conflicts: dict[str, list[str]] = {}
    for node_id, anchor, there, here in inherited_conflicts(name):
        conflicts.setdefault(node_id, []).append(f'{anchor[4:]} {there!r} -> {here!r}')
    replaced = {n['id'] for n in src.get('nodes') or [] if 'patch' not in n}
    patched = {n['id'] for n in src.get('nodes') or [] if 'patch' in n}
    firm, weak, total = [], [], 0
    for node in diagram.load(name)['nodes']:
        if node['id'] in replaced:
            continue
        total += 1
        why, deep = [], []
        if node['id'] in conflicts:
            why.append('reads a changed value: ' + '; '.join(conflicts[node['id']]))
        anchors = node.get('anchors') or []
        hit = [a for a in anchors if any(a == s or a.startswith(s + '.') for s in swaps)]
        if hit:
            why.append('anchors the swapped code: ' + ', '.join(hit))
        for a in anchors:
            if a.startswith('cfg:') or a in hit:
                continue
            near = _reached(a, 2) & methods
            far = (_reached(a, 3) & methods) - near
            if near:
                why.append(f'{a.split("::")[-1]} calls {", ".join(sorted(near))}')
            if far:
                deep.append(f'{a.split("::")[-1]} reaches {", ".join(sorted(far))} three calls down')
        text = ' '.join([node.get('title', '')] + list(node.get('lines') or []))
        said = [t for t in terms if t in text]
        if said:
            why.append('mentions ' + ', '.join(said))
        status = 'patched' if node['id'] in patched else 'inherited'
        if why:
            firm.append(f'{status:<9} {node["id"]:<12} ' + ' | '.join(why + deep))
        elif deep:
            weak.append(f'{status:<9} {node["id"]:<12} ' + ' | '.join(deep))
    return (['FLAGGED: re-read these boxes against the variant route'] + firm
            + ['WEAK: only a call three levels down reaches the swapped code; check the box claims briefly'] + weak
            + [f'{len(firm)} flagged, {len(weak)} weak, of {total} inherited or patched boxes; the rest stay '
               f'inherited unless the extraction or the audit finds otherwise'])


# ---------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('workflow')
    ap.add_argument('--stamp', action='store_true', help='record the current state as audited')
    ap.add_argument('--diff', action='store_true', help="print each changed function's body diff since the stamp")
    ap.add_argument('--triage', action='store_true', help="a variant: flag the inherited boxes worth re-reading")
    args = ap.parse_args(argv)
    name = args.workflow
    src = diagram.read(name)

    if args.triage:
        print('\n'.join(triage(name)))
        return 0

    for line in route_mismatches(src.get('route', {})):
        print(f'ROUTE    {line}')
    conflicts = inherited_conflicts(name)
    for node_id, anchor, there, here in conflicts:
        print(f'ROUTE    {node_id:<12} inherited, reads {anchor} = {here!r} on this route, {there!r} on the base: '
              f'override the box')

    if args.stamp:
        state, missing = current_state(name)
        for node_id, anchor, why in missing:
            print(f'MISSING  {node_id:<12} {anchor}  ({why})')
        if missing or conflicts:
            print('not stamped: fix the missing anchors and override the boxes ROUTE names first')
            return 2
        files = sorted({a.split('::')[0] for a in state if '::' in a and not a.startswith('mxtaltools/')})
        dirty = [f for f in files if _git('status', '--porcelain', '--', f)]
        head = _head(ENERGY_ROOT)
        stamp = {'commit': head, 'mxt_commit': _head(MXT_ROOT), 'dirty': dirty,
                 'date': dt.date.today().isoformat(), 'anchors': state}
        (HERE / f'{name}.stamp.json').write_text(json.dumps(stamp, indent=1, sort_keys=True), encoding='utf-8')
        print(f'stamped {len(state)} anchors at {head}' + (f' (working tree differs in {", ".join(dirty)})' if dirty else ''))
        return 0

    lines, drift, _, missing = drift_report(name, args.diff)
    stamp_path = HERE / f'{name}.stamp.json'
    if stamp_path.exists():
        stamp = json.loads(stamp_path.read_text(encoding='utf-8'))
        print(f'{name}: stamp {stamp["commit"]} ({stamp["date"]}) -> HEAD {_head(ENERGY_ROOT)}')
    for line in lines:
        print(line)
    base_drift = 0
    if src.get('extends'):
        _, base_drift, base_nodes, base_missing = drift_report(src['extends'])
        if base_drift or base_missing:
            mine = sorted({n['id'] for n in src.get('nodes') or []} & set(base_nodes))
            print(f'BASE     {src["extends"]}: {base_drift} drifted, {base_missing} missing anchors '
                  f'(boxes {", ".join(base_nodes) or "-"}): refresh the base first'
                  + (f'; this variant patches or replaces {", ".join(mine)}: re-read its versions too' if mine else ''))
            base_drift += base_missing
    print(f'{drift} drifted, {missing} missing, of {len(current_state(name)[0]) + missing} own anchors')
    if missing or not stamp_path.exists():
        return 2
    return 1 if (drift or conflicts or base_drift) else 0


if __name__ == '__main__':
    sys.exit(main())
