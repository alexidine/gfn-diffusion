"""
No runtime code may gate behaviour on a config key that `utils._RETIRED_KEYS`
rejects.

WHY THIS EXISTS, as a measured instance rather than a hypothetical.
`train.py::_stash_z_cal_cache` read `getattr(cfg, 'enabled', False)` where `cfg`
was the `z_calibration` block. `z_calibration.enabled` is retired and hard-fails
at load, so it could never be truthy: the stash returned on every call, the
cache it fills was never populated, and `z_calibration_tick`'s
`mode == 'regression'` guard therefore returned unconditionally. `mode:
regression` was unreachable for as long as that key had been retired.

WHY THE CLASS DESERVES AN INVARIANT. The failure is SILENT IN THE DIRECTION OF
LOOKING FINE. A retired key read through `getattr` with a falsy default cannot
raise, and the load-time gate never sees it, because the key is absent from
every config by construction. The run logs `z_cal/p = 0.0` and no
`z_cal/train_rms` -- indistinguishable from a calibration that ran and had
nothing to do. Nothing reports that a whole mode is dead.

HOW A READ IS MATCHED: BY DOTTED PATH, NOT BY LEAF. Two earlier cuts of this
test failed in opposite directions, and both failures are why the resolver below
exists.

  * Leaf-only matching flagged `profiling.enabled` and `adaptive_lr.ray_calibration`
    -- live keys whose LEAF collides with a retired one. Half of `_RETIRED_KEYS`
    are relocations, so a leaf is routinely live at its new home.
  * Excluding the colliding leaves as "too generic" removed `enabled`, and with
    it the only defect the file was written for.

So paths are reconstructed: attribute chains directly, and a `getattr(cfg, ...)`
whose receiver is a local by resolving that local back to its assignment in the
same function.
"""

import ast
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[2]   # tests/<area>/x.py -> energy_sampling/

pytestmark = pytest.mark.fast

# Runtime modules. `utils.py` holds the table and `config_state.py` /
# `config_invariants.py` implement retirement and migration, so all three name
# retired keys as data rather than gating on them.
RUNTIME = ('train.py', 'protocol.py', 'controller.py', 'buffer.py',
           'checkpointing.py', 'gflownet_losses.py', 'ray_calibration.py',
           'lr_bracket.py', 'lr_bracket_probe.py', 'lr_larder.py',
           'grad_clip_guard.py', 'profiling.py')


def _retired_table():
    """{dotted key: reason} straight out of `utils._RETIRED_KEYS`."""
    tree = ast.parse((HERE / 'utils.py').read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and any(getattr(t, 'id', None) == '_RETIRED_KEYS' for t in node.targets)):
            out = {}
            for k, v in zip(node.value.keys, node.value.values):
                if isinstance(k, ast.Constant):
                    try:
                        out[k.value] = ast.literal_eval(v)
                    except (ValueError, SyntaxError):
                        out[k.value] = ''
            assert out, '_RETIRED_KEYS parsed empty -- the scan would pass vacuously'
            return out
    raise AssertionError('_RETIRED_KEYS not found in utils.py')


def _retired_keys():
    """Every retired dotted key -- relocations included.

    Relocations are NOT excluded, and this was the third correction the file
    needed. `config_state.py`'s migration gate fires when a CONFIG carries a
    moved key; it says nothing about CODE still reading the old path, which is
    precisely the `z_calibration.enabled` defect. Excluding relocations made the
    scan miss the only bug it was written for. What makes including them safe is
    the segment rule in `_matches`, not a filter here.
    """
    return set(_retired_table())


def _chain(node):
    """Dotted path of a plain attribute chain, else None.

    `self.args.buffers.replay_buffer.toxic_min_draws` -> that string.
    """
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    name = getattr(node, 'id', None)
    if name is None:
        return None
    parts.append(name)
    return '.'.join(reversed(parts))


def _local_bindings(fn):
    """Map locals bound to a config block back to their path.

    `cfg = getattr(self.args, 'z_calibration', None)` -> {'cfg': 'self.args.z_calibration'}
    `cfg = self.args.z_calibration`                   -> the same.

    Only those two shapes are tracked. Anything else leaves the name unbound,
    and a read through it simply does not resolve -- which is the conservative
    direction here, since an unresolved read is not reported as a hit.
    """
    out = {}
    for node in ast.walk(fn):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        tgt = getattr(node.targets[0], 'id', None)
        if tgt is None:
            continue
        v = node.value
        if (isinstance(v, ast.Call) and getattr(v.func, 'id', None) == 'getattr'
                and len(v.args) >= 2 and isinstance(v.args[1], ast.Constant)
                and isinstance(v.args[1].value, str)):
            base = _chain(v.args[0])
            if base:
                out[tgt] = f'{base}.{v.args[1].value}'
        elif isinstance(v, ast.Attribute):
            base = _chain(v)
            if base:
                out[tgt] = base
    return out


def _matches(path, retired):
    """True when the reconstructed path names a retired key.

    SUFFIX MATCH FOR MULTI-SEGMENT KEYS, because the read side carries a prefix
    (`self.args.`, `self.m.args.`) that the retirement table does not:
    `self.args.buffers.replay_buffer.toxic_min_draws` names
    `buffers.replay_buffer.toxic_min_draws`.

    ROOT-ONLY MATCH FOR SINGLE-SEGMENT KEYS. A bare leaf like `ray_calibration`
    was retired FROM THE TOP LEVEL and lives on at `adaptive_lr.ray_calibration`,
    so a plain suffix match flags the correct new read. A one-segment key is
    therefore only a hit when it sits directly under the config root -- which is
    exactly where it was retired from.
    """
    if not path:
        return False
    segs = path.split('.')
    for key in retired:
        ks = key.split('.')
        if len(ks) == 1:
            if len(segs) >= 2 and segs[-1] == ks[0] and segs[-2] == 'args':
                return True
        elif len(segs) >= len(ks) and segs[-len(ks):] == ks:
            return True
    return False


def _config_reads(src):
    """Every (dotted path, lineno) at which this source reads an attribute."""
    tree = ast.parse(src)
    reads = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        binds = _local_bindings(fn)
        for node in ast.walk(fn):
            if isinstance(node, ast.Call) and getattr(node.func, 'id', None) == 'getattr':
                if len(node.args) < 2 or not isinstance(node.args[1], ast.Constant):
                    continue
                leaf = node.args[1].value
                if not isinstance(leaf, str):
                    continue
                base = _chain(node.args[0])
                base = binds.get(base, base)
                if base:
                    reads.append((f'{base}.{leaf}', node.lineno))
            elif isinstance(node, ast.Attribute):
                p = _chain(node)
                if p:
                    reads.append((binds.get(p, p), node.lineno))
    return reads


def test_the_retirement_table_is_readable_and_nonempty():
    """The scan is only as good as this list. An empty or unparseable table
    would make every other test here pass while checking nothing."""
    retired = _retired_keys()
    assert len(retired) >= 5, f'only {len(retired)} retirements parsed'
    assert "buffers.replay_buffer.toxic_min_draws" in retired


@pytest.mark.parametrize('name', RUNTIME)
def test_no_runtime_module_gates_on_a_retired_key(name):
    path = HERE / name
    assert path.exists(), f'{name} is named here but is not in the tree'
    retired = _retired_keys()
    hits = sorted({(p, ln) for p, ln in _config_reads(path.read_text(encoding='utf-8'))
                   if _matches(p, retired)})
    assert not hits, (
        f'{name} reads retired config key(s): '
        + ', '.join(f'{p} at line {ln}' for p, ln in hits)
        + '. A retired key can never be set, so the read always yields its '
          'default and the branch behind it is unreachable.')


def test_the_two_argument_getattr_shape_is_detected():
    """Mutation for the scan above. The real `_stash_z_cal_cache` shape --
    a local bound to a config block, then read through getattr -- must resolve
    to the retired path, or the module test passes because nothing matched."""
    src = ('def f(self):\n'
           "    cfg = getattr(self.args.buffers, 'replay_buffer', None)\n"
           "    if not getattr(cfg, 'toxic_min_draws', 0):\n"
           '        return\n')
    hits = [p for p, _ in _config_reads(src)
            if _matches(p, {'buffers.replay_buffer.toxic_min_draws'})]
    assert hits, 'the resolver did not reconstruct the retired path'


def test_the_plain_attribute_shape_is_detected():
    """The other spelling: no local, just the full chain."""
    src = ('def f(self):\n'
           '    return self.args.buffers.replay_buffer.toxic_min_draws\n')
    hits = [p for p, _ in _config_reads(src)
            if _matches(p, {'buffers.replay_buffer.toxic_min_draws'})]
    assert hits, 'a plain attribute chain is not resolved'


def test_a_live_key_whose_leaf_collides_is_not_flagged():
    """The false positive that broke the leaf-only version. `profiling.enabled`
    is live, `z_calibration.enabled` is retired: same leaf, different verdict."""
    src = 'def f(self):\n    return self.args.profiling.enabled\n'
    reads = _config_reads(src)
    assert reads, 'the resolver found no read at all'
    assert not [p for p, _ in reads if _matches(p, {'z_calibration.enabled'})]


def test_an_objects_own_attribute_is_not_a_config_read():
    """`self.enabled` on RayCalibration is instance state, not config. Matching
    it would flood the scan with the noise that motivated the bad exclusion."""
    src = 'def f(self):\n    return self.enabled\n'
    assert not [p for p, _ in _config_reads(src)
                if _matches(p, {'z_calibration.enabled'})]
