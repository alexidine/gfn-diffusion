"""Reference checker for the wiki pages under docs/wiki/.

The writing protocol promises that the three reference notations a page uses are
"checked by a script". This is that script. It reads every page except index.md
and writing-protocol.md, pulls the references out of the inline code spans, and
resolves each one against the code and the canonical config.

    python docs/wiki/check_refs.py                 # check everything, exit 1 on failure
    python docs/wiki/check_refs.py --page prior-buffer
    python docs/wiki/check_refs.py --drift         # report-only: what moved since each stamp

THREE KINDS OF REFERENCE, all inside single-backtick code spans:

  symbol   `train.py::Modeller.train`, `models/gfn.py::GFN._fwd_step`,
           `utils.py::_RETIRED_KEYS`; then `::Name` to continue in the last file
           and class named, and `.name` to continue a run of spans on the same
           line. A path beginning `mxtaltools/` resolves in the sibling
           repository; every other path in energy_sampling. Pages write the full
           path in `Sources` and the readable tail in the body, so a tail is
           matched against the page's own full paths, then against both trees.
  config   `cfg:a.b.c`, the brace product `cfg:a.{x, y}` and `cfg:{x,y}_suffix`,
           and `cfg:stage.<key>`, which means a key declared by SOME stage block
           of SOME protocol in the canonical config. Resolved against
           configs/mk_dev.yaml, then the two conformer configs, then -- for the
           three pages about a program that is not the trainer -- that program's
           own config. A key no config carries is resolved against the trainer's
           own vocabulary, since several blocks document keys that appear in no
           config and take the code default; the block it sits under must still
           exist, so an invented block is still a failure.
  page     `[name](name.md)`, which must name a file in docs/wiki/.

Resolution is by `ast` and `yaml` only.  NOTHING HERE IMPORTS TORCH, or any
module of the trainer: the point is a check that runs in the fast test tier, on
a machine with no GPU, in under a second.  `tests/wiki/test_wiki_refs.py` pins
that.

Deliberately unresolvable references -- a symbol a page names BECAUSE it was
deleted, an environment variable -- go in check_refs_allow.txt, one per line
with a trailing `# reason`.
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

WIKI = Path(__file__).resolve().parent
ENERGY_ROOT = WIKI.parents[1]                      # .../gfn_diffusion/energy_sampling
ENERGY_GIT = ENERGY_ROOT.parent                    # .../gfn_diffusion  (the repo root)
MXT_ROOT = ENERGY_ROOT.parents[1] / 'mxtaltools'   # .../mxt_gfn/mxtaltools (repo root)

#: index.md is a table of contents and writing-protocol.md is the notation's own
#: definition; both cite the notation rather than using it.
SKIP_PAGES = {'index.md', 'writing-protocol.md'}

CONFIG_FILES = ('configs/mk_dev.yaml',)
#: Conformer pages cite the conformer configs, which mk_dev does not carry.
FALLBACK_CONFIG_FILES = ('configs/conformer_dev.yaml', 'configs/conformer_mk.yaml')
#: Three pages are about programs that are not the trainer and read a config of
#: their own, which each says by name: the crystal search and the paper-figure
#: analysis. Their keys are matched on any dotted suffix, because those files
#: nest per-run blocks (`base.std_orientation`) that the pages address flat.
AUX_CONFIG_FILES = (ENERGY_ROOT / 'eval/paper1_results/new_analysis.yaml',
                    MXT_ROOT / 'configs/crystal_searches/base.yaml')

ALLOW_FILE = WIKI / 'check_refs_allow.txt'

# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------

#: A single-backtick code span. Double-backtick spans are not used on any page.
_SPAN = re.compile(r'`([^`\n]+)`')
_FENCE = re.compile(r'^\s*```')
_LINK = re.compile(r'\[[^\]]*\]\(([^)]+)\)')
_DRIFT = re.compile(r'Verified against commit `([0-9a-f]{7,40})`')

#: `path/to/file.py::Sym` or `::Sym`, where Sym is `name` or `Class.name`.
_SYMBOL = re.compile(
    r'(?P<path>[A-Za-z0-9_][A-Za-z0-9_/.-]*\.py)?'
    r'::(?P<sym>[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?)')

#: A span that is nothing but `.name`: the continuation form used in the
#: `Sources` lists and in parenthetical runs. A call suffix is NOT allowed --
#: every `.name(...)` span on the pages today is prose about a call (`.long()`,
#: `.update(self.args.energy_config.__dict__)`), not a reference.
_CONTINUATION = re.compile(r'^\.([A-Za-z_][A-Za-z0-9_]*)$')

#: What may sit between the previous code span and a `.name` continuation for
#: the two to be one run: list punctuation and the two words the pages join a
#: pair with. Anything else and the `.name` is prose about an attribute
#: (`must carry a `.scalar` attribute`), which is not a reference to check.
_RUN_GAP = re.compile(r'^[\s,;:()\[\]]*(?:and|or)?[\s,;:()\[\]]*$')

#: `.py` and `.pt` are file extensions in prose, not continuations. They are the
#: only two that collide; keeping the list short keeps the collision visible.
_EXTENSIONS = {'py', 'pt', 'yaml', 'yml', 'md', 'json', 'txt', 'sh', 'csv'}

#: `cfg:` followed by a dotted key, with `{a, b}` products allowed in any
#: segment and `*` allowed as a wildcard segment character.
#: Commas and spaces are allowed only INSIDE a brace product: outside one they
#: end the key, or `max(3, cfg:max_reloads_per_1k_steps * step_ind / 1000)`
#: reads as a key called "max_reloads_per_1k_steps * step_ind".
_CFG = re.compile(r'cfg:(?P<key>(?:\{[^}]*\}|[A-Za-z0-9_*])'
                  r'(?:\{[^}]*\}|[A-Za-z0-9_*.])*)')
_BRACE = re.compile(r'\{([^{}]*)\}')


@dataclass(frozen=True)
class Ref:
    kind: str          # 'symbol' | 'cfg' | 'link'
    text: str          # the page's own spelling, and the allowlist key
    page: str
    line: int
    #: For a symbol ref, the symbols to try, in order. A continuation inside a
    #: class (`::compute_loss` after `::MolCrystalAnalysis.analyze`) is written
    #: for the class it sits in, but the same notation continues a module-level
    #: run too, so both spellings are candidates and either one resolving is a
    #: resolved reference.
    cands: tuple[str, ...] = ()
    #: For a symbol ref, the files to try, in order. A `::Name` continuation
    #: names the last file for a reader who is following the paragraph, and
    #: pages interleave files inside one paragraph (`::GFN.split_params` a page
    #: after gfn.py was last spelled out), so the other files the SAME page
    #: cites are candidates behind it.
    paths: tuple[str, ...] = ()
    #: True for a continuation, which names no file and so no scope either: the
    #: page is inside a class and the reader knows it. A bare name in that
    #: position is allowed to be a member of one of the file's classes as well
    #: as a module-level name.
    loose: bool = False


def _expand_braces(key: str) -> list[str]:
    """`a.{x, y}_z` -> ['a.x_z', 'a.y_z']. Products compose left to right."""
    out = [key]
    while True:
        grown = []
        changed = False
        for k in out:
            m = _BRACE.search(k)
            if not m:
                grown.append(k)
                continue
            changed = True
            for alt in m.group(1).split(','):
                grown.append(k[:m.start()] + alt.strip() + k[m.end():])
        out = grown
        if not changed:
            return out


def extract(page: Path, is_class=None, page_paths=()) -> list[Ref]:
    """Every reference on one page, in document order.

    `is_class(path, name)` decides whether a bare `::Name` named a class, which
    is what a following `.member` span continues into. Without it every
    continuation is read as module level, which is enough to collect the FILES a
    page refers to (what --drift needs) and not enough to resolve members."""
    refs: list[Ref] = []
    name = page.name
    last_file: str | None = None
    last_class: str | None = None
    in_fence = False

    for lineno, line in enumerate(page.read_text(encoding='utf-8').splitlines(), 1):
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        for target in _LINK.findall(line):
            if target.endswith('.md') and '/' not in target and not target.startswith('#'):
                refs.append(Ref('link', target, name, lineno))

        # `.name` continues the last symbol named on THIS line; `::Name`
        # continues the last file and class named, which may be lines back.
        line_class: str | None = None
        line_file: str | None = None

        prev_end = -1
        for sm in _SPAN.finditer(line):
            span, gap = sm.group(1), line[prev_end:sm.start()] if prev_end >= 0 else None
            prev_end = sm.end()
            for m in _CFG.finditer(span):
                key = m.group('key').strip().rstrip('.').strip()
                if key:
                    refs.append(Ref('cfg', 'cfg:' + key, name, lineno))

            hits = list(_SYMBOL.finditer(span))
            for m in hits:
                if m.group('path'):
                    last_file, last_class = m.group('path'), None
                path = m.group('path') or last_file
                if not path:
                    continue
                line_file = path
                paths = ((path,) if m.group('path')
                         else (path,) + tuple(p for p in page_paths if p != path))
                sym = m.group('sym')
                if '.' in sym:
                    last_class = sym.split('.')[0]
                    cands = (sym,)
                elif is_class is not None and is_class(path, sym):
                    last_class, cands = sym, (sym,)
                elif last_class:
                    cands = (f'{last_class}.{sym}', sym)
                else:
                    cands = (sym,)
                line_class = last_class
                refs.append(Ref('symbol', f'{path}::{sym}', name, lineno, cands,
                                paths, not m.group('path')))

            if hits:
                continue
            cont = _CONTINUATION.match(span.strip())
            if (cont and line_file and cont.group(1) not in _EXTENSIONS
                    and gap is not None and _RUN_GAP.match(gap)):
                sym = cont.group(1)
                cands = (f'{line_class}.{sym}', sym) if line_class else (sym,)
                paths = (line_file,) + tuple(p for p in page_paths if p != line_file)
                refs.append(Ref('symbol', f'{line_file}::{sym}', name, lineno,
                                cands, paths, True))

    return refs


# --------------------------------------------------------------------------
# symbol resolution, by AST
# --------------------------------------------------------------------------

@dataclass
class ModuleIndex:
    names: set[str] = field(default_factory=set)               # module level
    classes: dict[str, set[str]] = field(default_factory=dict)  # class -> members
    bases: dict[str, list[str]] = field(default_factory=dict)   # class -> base names


def _bound_names(node: ast.AST) -> list[str]:
    """Names a statement binds at the level it appears."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [node.name]
    if isinstance(node, ast.Assign):
        out = []
        for t in node.targets:
            if isinstance(t, ast.Name):
                out.append(t.id)
            elif isinstance(t, (ast.Tuple, ast.List)):
                out += [e.id for e in t.elts if isinstance(e, ast.Name)]
        return out
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return [node.target.id]
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        # An imported name IS resolvable from the module a page names it on, and
        # several pages cite re-exports that way.
        return [(a.asname or a.name).split('.')[0] for a in node.names]
    return []


def _index_module(path: Path) -> ModuleIndex:
    idx = ModuleIndex()
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))

    def walk_body(body, sink: set[str]):
        for node in body:
            sink.update(_bound_names(node))
            # `if TYPE_CHECKING:` / `try: ... except ImportError:` still bind.
            for attr in ('body', 'orelse', 'finalbody'):
                inner = getattr(node, attr, None)
                if inner and isinstance(node, (ast.If, ast.Try)):
                    walk_body(inner, sink)
            for handler in getattr(node, 'handlers', []) or []:
                walk_body(handler.body, sink)

    walk_body(tree.body, idx.names)

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            members: set[str] = set()
            walk_body(node.body, members)
            # `self.x = ...` inside a method is a member too, and several pages
            # cite one (`buffer.py::ConditionLogZTracker.ema_logw`).
            for inner in ast.walk(node):
                if isinstance(inner, (ast.Assign, ast.AnnAssign)):
                    targets = (inner.targets if isinstance(inner, ast.Assign)
                               else [inner.target])
                    for t in targets:
                        if (isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                                and t.value.id == 'self'):
                            members.add(t.attr)
            idx.classes[node.name] = members
            idx.bases[node.name] = [b.id for b in node.bases if isinstance(b, ast.Name)]
    return idx


#: Directories that hold no source a page cites, and that are large enough that
#: walking them for the short-path fallback is the slow part of the run.
_SKIP_DIRS = {'.git', '__pycache__', 'wandb', 'checkpoints', 'SCRATCH', '.claude',
              'node_modules', '.idea', 'site'}


def _walk_py(root: Path) -> list[str]:
    out = []
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            entries = list(d.iterdir())
        except OSError:
            continue
        for e in entries:
            if e.is_dir():
                if e.name not in _SKIP_DIRS:
                    stack.append(e)
            elif e.suffix == '.py':
                out.append(e.relative_to(root).as_posix())
    return out


class CodeIndex:
    """Resolves a cited path to a file, then a symbol inside it.

    A page writes the full path in its `Sources` line (`mxtaltools/dataset_utils/
    data_class_methods/crystal_ops.py`) and the readable tail in the body
    (`crystal_ops.py`). Both are the same reference, so a tail that is not a file
    on its own is matched against the full paths cited on the SAME page first,
    and only then against the two repositories, where it has to be unique."""

    def __init__(self):
        self._cache: dict[str, ModuleIndex | None] = {}
        self._files: dict[str, list[tuple[str, Path]]] | None = None
        self.hints: set[str] = set()

    # -- path ------------------------------------------------------------
    def _all_files(self):
        if self._files is None:
            self._files = {}
            for root in (ENERGY_ROOT, MXT_ROOT):
                for rel in _walk_py(root):
                    self._files.setdefault(rel.rsplit('/', 1)[-1], []).append((rel, root))
        return self._files

    def files_for(self, path: str) -> list[Path]:
        """Every file the cited path could name, best first.

        More than one is not an error to report: two files of the same name in
        two packages are disambiguated by which one holds the symbol, so the
        caller tries them in turn."""
        for root in ((MXT_ROOT, ENERGY_ROOT) if path.startswith('mxtaltools/')
                     else (ENERGY_ROOT, MXT_ROOT)):
            if (root / path).exists():
                return [root / path]
        tail = '/' + path
        local = sorted({h for h in self.hints if h.endswith(tail)})
        if local:
            return [f for h in local for f in self.files_for(h)]
        return [root / rel
                for rel, root in self._all_files().get(path.rsplit('/', 1)[-1], [])
                if rel == path or rel.endswith(tail)]

    def file_for(self, path: str) -> Path | None:
        found = self.files_for(path)
        return found[0] if found else None

    def get(self, path: str, f: Path | None = None) -> ModuleIndex | None:
        f = f or self.file_for(path)
        key = str(f)
        if key not in self._cache:
            try:
                self._cache[key] = _index_module(f)
            except (OSError, SyntaxError, TypeError, AttributeError, ValueError):
                self._cache[key] = None
        return self._cache[key]

    def is_class(self, path: str, name: str) -> bool:
        idx = self.get(path)
        return bool(idx and name in idx.classes)

    def has_member(self, idx: ModuleIndex, cls: str, member: str, seen=()) -> bool:
        if cls in seen:
            return False
        members = idx.classes.get(cls)
        if members is None:
            return False
        if member in members:
            return True
        return any(self.has_member(idx, b, member, seen + (cls,))
                   for b in idx.bases.get(cls, []))

    def resolve(self, ref: Ref) -> str:
        """'' when the reference resolves, else why it did not."""
        first = ref.text.split('::', 1)[0]
        why = ''
        for i, path in enumerate(ref.paths or (first,)):
            files = self.files_for(path)
            if not files:
                if i == 0:
                    why = f'no such file: {path}'
                continue
            for f in files:
                reason = self._in_file(ref, path, f)
                if not reason:
                    return ''
                if i == 0:
                    why = reason
        return why or f'no such file: {first}'

    def _in_file(self, ref: Ref, path: str, f: Path) -> str:
        idx = self.get(path, f)
        if idx is None:
            return f'could not parse {path}'
        why = ''
        for sym in ref.cands or (ref.text.split('::', 1)[1],):
            if '.' in sym:
                cls, member = sym.split('.', 1)
                if cls not in idx.classes:
                    # A module-level object with an attribute: the attribute is
                    # not in the AST, and the object being there is the claim.
                    why = '' if cls in idx.names else f'no class {cls} in {path}'
                elif not self.has_member(idx, cls, member):
                    why = f'{cls} has no member {member} in {path}'
                else:
                    why = ''
            elif sym in idx.names or sym in idx.classes:
                why = ''
            elif ref.loose and any(sym in m for m in idx.classes.values()):
                why = ''
            else:
                why = f'no module-level {sym} in {path}'
            if not why:
                return ''
        return why


# --------------------------------------------------------------------------
# config resolution
# --------------------------------------------------------------------------

def _flatten(obj, prefix='') -> set[str]:
    """Every dotted key path in a config. A list of blocks (`opt` in the search
    config is a list of stages) contributes its elements' keys at the list's own
    path, since that is how a page addresses them."""
    out = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f'{prefix}{k}'
            out.add(key)
            out |= _flatten(v, key + '.')
    elif isinstance(obj, list):
        for item in obj:
            out |= _flatten(item, prefix)
    return out


def _string_constants(path: Path) -> set[str]:
    """Every identifier-shaped string literal in a module."""
    try:
        tree = ast.parse(path.read_text(encoding='utf-8'))
    except (OSError, SyntaxError):
        return set()
    return {n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and n.value.isidentifier()}


def _config_vocabulary(root: Path) -> set[str]:
    """Names the trainer reads a config value by: `args.x`, `cfg['x']`, `x=`.

    A key the canonical configs do not carry is not necessarily a mistake --
    several blocks document keys that "appear in no config" and take the code
    default -- but a key NOTHING names is. This is the vocabulary such a key has
    to be in."""
    out: set[str] = set()
    for rel in ('*.py', 'energies/*.py', 'models/*.py'):
        for f in root.glob(rel):
            try:
                tree = ast.parse(f.read_text(encoding='utf-8'))
            except (OSError, SyntaxError, UnicodeDecodeError):
                continue
            for n in ast.walk(tree):
                if isinstance(n, ast.Attribute):
                    out.add(n.attr)
                elif isinstance(n, ast.Constant) and isinstance(n.value, str):
                    if n.value.isidentifier():
                        out.add(n.value)
                elif isinstance(n, ast.keyword) and n.arg:
                    out.add(n.arg)
                elif isinstance(n, ast.arg):
                    out.add(n.arg)
    return out


def _suffixes(keys: set[str]) -> set[str]:
    out = set()
    for k in keys:
        parts = k.split('.')
        out |= {'.'.join(parts[i:]) for i in range(len(parts))}
    return out


class ConfigIndex:
    def __init__(self, root: Path = ENERGY_ROOT):
        self.keys: set[str] = set()
        self.aux_keys: set[str] = set()
        self.stage_keys: set[str] = set()
        self.protocols: set[str] = set()
        for rel in CONFIG_FILES + FALLBACK_CONFIG_FILES:
            self._load(root / rel, stages=True)
        for path in AUX_CONFIG_FILES:
            before = set(self.keys)
            self._load(path, stages=False)
            self.aux_keys |= _suffixes(self.keys - before)
        #: A stage block declares what the stage PARSER accepts, and mk_dev
        #: exercises a part of that vocabulary: the controller kinds it does not
        #: run declare keys no shipped stage carries. `protocol.py`'s own string
        #: constants are the rest of the vocabulary.
        self.stage_vocab = _string_constants(root / 'protocol.py')
        self.code_vocab = _config_vocabulary(root)

    def _load(self, path: Path, stages: bool):
        try:
            raw = yaml.safe_load(path.read_text(encoding='utf-8'))
        except (OSError, yaml.YAMLError):
            return
        if not isinstance(raw, dict):
            return
        self.keys |= _flatten(raw)
        if not stages:
            return
        for name, proto in (raw.get('protocols') or {}).items():
            self.protocols.add(name)
            for stage in (proto or {}).get('stages') or []:
                if isinstance(stage, dict):
                    self.stage_keys |= _flatten(stage)

    @staticmethod
    def _matches(key: str, pool: set[str]) -> bool:
        if '*' not in key:
            return key in pool
        pat = re.compile('^' + '[^.]*'.join(re.escape(p) for p in key.split('*')) + '$')
        return any(pat.match(k) for k in pool)

    def _code_default(self, key: str) -> bool:
        """A key the configs omit because the code default stands.

        The block it sits under must still be a block the configs declare, so
        this cannot pass a key invented under a block that does not exist."""
        parent, _, leaf = key.rpartition('.')
        if parent and not self._matches(parent, self.keys):
            return False
        return leaf.replace('*', '') in self.code_vocab

    def _stage(self, key: str) -> bool:
        """A stage key: declared by a shipped stage, or named by the parser."""
        return (self._matches(key, self.stage_keys)
                or all(p.replace('*', '') in self.stage_vocab for p in key.split('.')))

    def resolve(self, ref: Ref) -> str:
        bad = []
        for key in _expand_braces(ref.text[len('cfg:'):]):
            key = key.strip().strip('.')
            if not key:
                continue
            # `protocols.<name>.stages[<tag>].<rest>` addresses a stage block.
            m = re.match(r'protocols\.([A-Za-z0-9_*]+)\.stages\[[^\]]*\]\.?(.*)$', key)
            if m:
                rest = m.group(2)
                if not rest or self._stage(rest):
                    continue
                bad.append(key)
                continue
            if key.startswith('stage.'):
                if self._stage(key[len('stage.'):]):
                    continue
                bad.append(key)
                continue
            if not (self._matches(key, self.keys)
                    or self._matches(key, self.aux_keys)
                    or self._code_default(key)):
                bad.append(key)
        return 'absent from the configs: ' + ', '.join(bad) if bad else ''


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def load_allowlist(path: Path = ALLOW_FILE) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        ref, _, reason = line.partition('#')
        out[ref.strip()] = reason.strip()
    return out


def pages(only: str | None = None) -> list[Path]:
    out = [p for p in sorted(WIKI.glob('*.md')) if p.name not in SKIP_PAGES]
    if only:
        want = only if only.endswith('.md') else only + '.md'
        out = [p for p in out if p.name == want]
    return out


def check(only: str | None = None):
    """Returns (failures, counts). A failure is (Ref, reason)."""
    code, cfg = CodeIndex(), ConfigIndex()
    allow = load_allowlist()
    known = {p.name for p in WIKI.glob('*.md')}
    failures, counts = [], {'symbol': 0, 'cfg': 0, 'link': 0, 'allowlisted': 0}

    for page in pages(only):
        # First pass: the full paths this page cites, which the short forms in
        # its body are then matched against. Second pass: the refs themselves.
        cited = {r.text.split('::', 1)[0] for r in extract(page) if r.kind == 'symbol'}
        code.hints = {h for h in cited
                      if (MXT_ROOT / h).exists() or (ENERGY_ROOT / h).exists()}
        for ref in extract(page, is_class=code.is_class, page_paths=sorted(cited)):
            counts[ref.kind] += 1
            if ref.kind == 'symbol':
                why = code.resolve(ref)
            elif ref.kind == 'cfg':
                why = cfg.resolve(ref)
            else:
                why = '' if ref.text in known else 'no such page'
            if not why:
                continue
            if ref.text in allow:
                counts['allowlisted'] += 1
                continue
            failures.append((ref, why))
    return failures, counts


def drift(only: str | None = None) -> list[str]:
    """One report line per page: stamp, files referenced, files changed since.

    A REPORT, never a verdict. A page cites files it describes and files it only
    mentions, and a commit touching one is a reason to re-read the page, not
    evidence that the page is wrong."""
    lines = []
    code = CodeIndex()
    for page in pages(only):
        text = page.read_text(encoding='utf-8')
        m = _DRIFT.search(text)
        if not m:
            lines.append(f'{page.name}: no drift stamp')
            continue
        stamp = m.group(1)
        # The page's short spellings (`crystal_ops.py`) resolve to real files
        # first: a path git does not know is silently never changed.
        code.hints = {r.text.split('::', 1)[0] for r in extract(page)
                      if r.kind == 'symbol'}
        files = sorted({f for cited in code.hints for f in code.files_for(cited)})
        changed = []
        for f in files:
            if MXT_ROOT in f.parents:
                # The stamp is a commit of the OTHER repository, so the sibling
                # is read by the stamp's date instead.
                repo = MXT_ROOT
                arg = ['--since', _stamp_date(stamp)]
                rel = f.relative_to(MXT_ROOT).as_posix()
            else:
                repo, arg = ENERGY_GIT, [f'{stamp}..HEAD']
                rel = f.relative_to(ENERGY_GIT).as_posix()
            if _git(repo, ['log', '--oneline', *arg, '--', rel]):
                changed.append(rel)
        lines.append(f'{page.name}: {stamp}  {len(files)} files  '
                     f'{len(changed)} changed'
                     + (': ' + ', '.join(changed) if changed else ''))
    return lines


def _git(repo: Path, args: list[str]) -> str:
    try:
        return subprocess.run(['git', '-C', str(repo), *args], capture_output=True,
                              text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ''


def _stamp_date(stamp: str) -> str:
    return _git(ENERGY_GIT, ['log', '-1', '--format=%cI', stamp]) or '1970-01-01'


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--drift', action='store_true',
                    help='report what moved since each page stamp; never fails')
    ap.add_argument('--page', help='one page, with or without .md')
    args = ap.parse_args(argv)

    if args.drift:
        for line in drift(args.page):
            print(line)
        return 0

    failures, counts = check(args.page)
    print(f"{counts['symbol']} symbol, {counts['cfg']} config, {counts['link']} link "
          f"references; {counts['allowlisted']} allowlisted; {len(failures)} unresolved")
    for ref, why in failures:
        print(f'  {ref.page}:{ref.line}  {ref.text}  --  {why}')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
