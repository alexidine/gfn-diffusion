"""Change scanner for the workflow diagrams: what moved under a diagram since it was last audited.

    python docs/workvis/scan.py unconditional            # report against unconditional.stamp.json
    python docs/workvis/scan.py unconditional --stamp    # record the current state (after an audit)

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
value. A scan reports, by node: CHANGED (body hash moved), CALLS (a call added or
dropped -- the usual sign of a new branch the diagram may not show), CONFIG (value
moved), MISSING (anchor no longer resolves), and ROUTE (the committed config no longer
carries a value the route pins, so the diagram describes a config that is not the
committed one; or it now carries an override, which can become a pin). Exit 0 clean,
1 drift, 2 missing anchors or no stamp.

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
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ENERGY_ROOT = HERE.parents[1]                      # .../gfn_diffusion/energy_sampling
MXT_ROOT = ENERGY_ROOT.parents[1] / 'mxtaltools'   # .../mxt_gfn/mxtaltools

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


def resolve_symbol(anchor: str) -> ast.AST:
    rel, _, qual = anchor.partition('::')
    path = _file_for(rel)
    if not path.exists():
        raise LookupError(f'no file {rel}')
    node: ast.AST = _tree(path)
    for part in qual.split('.'):
        found = _find(getattr(node, 'body', []), part)
        if found is None:
            raise LookupError(f'{qual}: no {part!r} in {rel}')
        node = found
    return node


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


def fingerprint(anchor: str) -> dict:
    node = copy.deepcopy(resolve_symbol(anchor))
    body = getattr(node, 'body', None)
    if (isinstance(body, list) and body and isinstance(body[0], ast.Expr)
            and isinstance(getattr(body[0], 'value', None), ast.Constant)
            and isinstance(body[0].value.value, str)):
        node.body = body[1:] or [ast.Pass()]
    digest = hashlib.sha1(ast.dump(node, include_attributes=False).encode()).hexdigest()[:12]
    return {'hash': digest, 'calls': _calls(node)}


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


def route_config(src: dict) -> tuple[dict, dict]:
    """The committed config, and the route's config: the committed one with route.overrides set."""
    route = src.get('route', {})
    committed = yaml.safe_load((ENERGY_ROOT / route.get('config', 'configs/mk_dev.yaml')).read_text(encoding='utf-8'))
    cfg = copy.deepcopy(committed)
    for key, value in (route.get('overrides') or {}).items():
        *parents, leaf = key.split('.')
        node = cfg
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = value
    return committed, cfg


def _git(*args: str) -> str:
    try:
        return subprocess.run(['git', *args], cwd=ENERGY_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ''


def current_state(src: dict) -> tuple[dict, list[tuple[str, str, str]]]:
    """{anchor: fingerprint or {'value': ...}}, and the anchors that failed to resolve."""
    route = src.get('route', {})
    _, cfg = route_config(src)
    protocol = (route.get('pins') or {}).get('protocol') or cfg.get('protocol')
    state, missing = {}, []
    for node in src['nodes']:
        for anchor in node.get('anchors') or []:
            if anchor in state:
                continue
            try:
                if anchor.startswith('cfg:'):
                    state[anchor] = {'value': resolve_config(anchor, cfg, protocol)}
                else:
                    state[anchor] = fingerprint(anchor)
            except (LookupError, SyntaxError) as e:
                missing.append((node['id'], anchor, str(e)))
    return state, missing


def route_mismatches(src: dict) -> list[str]:
    route = src.get('route', {})
    cfg, _ = route_config(src)
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('workflow')
    ap.add_argument('--stamp', action='store_true', help='record the current state as audited')
    args = ap.parse_args(argv)

    src = yaml.safe_load((HERE / f'{args.workflow}.yaml').read_text(encoding='utf-8'))
    stamp_path = HERE / f'{args.workflow}.stamp.json'
    state, missing = current_state(src)
    head = _git('rev-parse', '--short', 'HEAD')
    files = sorted({a.split('::')[0] for a in state if '::' in a and not a.startswith('mxtaltools/')})
    dirty = [f for f in files if _git('status', '--porcelain', '--', f)]

    for node_id, anchor, why in missing:
        print(f'MISSING  {node_id:<12} {anchor}  ({why})')
    for line in route_mismatches(src):
        print(f'ROUTE    {line}')

    if args.stamp:
        if missing:
            print('not stamped: fix the missing anchors first')
            return 2
        stamp = {'commit': head, 'dirty': dirty, 'date': dt.date.today().isoformat(), 'anchors': state}
        stamp_path.write_text(json.dumps(stamp, indent=1, sort_keys=True), encoding='utf-8')
        print(f'stamped {len(state)} anchors at {head}' + (f' (working tree differs in {", ".join(dirty)})' if dirty else ''))
        return 0

    if not stamp_path.exists():
        print(f'no stamp for {args.workflow}: audit the diagram, then run with --stamp')
        return 2
    stamp = json.loads(stamp_path.read_text(encoding='utf-8'))
    print(f'{args.workflow}: stamp {stamp["commit"]} ({stamp["date"]}) -> HEAD {head}'
          + (f', working tree differs in {", ".join(dirty)}' if dirty else ''))
    owners: dict[str, list[str]] = {}
    for node in src['nodes']:
        for anchor in node.get('anchors') or []:
            owners.setdefault(anchor, []).append(node['id'])
    drift = 0
    for anchor, now in state.items():
        then = stamp['anchors'].get(anchor)
        who = ','.join(owners.get(anchor, []))
        if then is None:
            print(f'NEW      {who:<12} {anchor}  (not in the stamp)')
            drift += 1
        elif 'value' in now:
            if now['value'] != then.get('value'):
                print(f'CONFIG   {who:<12} {anchor}  {then.get("value")!r} -> {now["value"]!r}')
                drift += 1
        elif now['hash'] != then.get('hash'):
            added = sorted(set(now['calls']) - set(then.get('calls', [])))
            dropped = sorted(set(then.get('calls', [])) - set(now['calls']))
            print(f'CHANGED  {who:<12} {anchor}')
            if added or dropped:
                print(f'  CALLS  +{", ".join(added) or "none"}  -{", ".join(dropped) or "none"}')
            drift += 1
    print(f'{drift} drifted, {len(missing)} missing, of {len(state) + len(missing)} anchors')
    return 2 if missing else (1 if drift else 0)


if __name__ == '__main__':
    sys.exit(main())
