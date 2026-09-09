"""A stage action a config can name must be callable on the routes that can name it.

`bootstrap_z` was listed in `protocol.ACTIONS`, documented as THE way to anchor log Z at a
stage boundary, reachable from any config, and BROKEN ON EVERY ROUTE for as long as the
signature it depended on had been current. Its `_draw_batch` unpacked six values from
`energy_function.condition_samples`; both energies return four. `MolecularCrystal`'s own
comment says the crystal-shaped members were dropped "so a non-crystal energy_function can
implement it without padding" -- and this one caller was never moved with them.

Nothing caught it because nothing exercised it: no test names any stage action, and the only
signal was an exception thrown deep inside a stage transition, at a point most runs never
reach. An action that is listed and documented and raises is worse than one that is absent,
because a config that asks for it reads as protected.

THESE TESTS ARE ABOUT CONTRACTS, NOT BEHAVIOUR. They do not assert that `bootstrap_z` anchors
log Z well -- that is a training question. They assert that the arity every caller depends on
is the arity the implementations provide, which is the part that was silently false.
"""
import ast
import inspect
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _return_arity(path, class_name, method):
    """How many values `class_name.method` returns, by parsing rather than importing.

    Parsed because importing `train` drags in torch, wandb and the whole modeller; the
    question here is purely syntactic and should stay cheap enough to run everywhere.
    """
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ClassDef) and node.name == class_name):
            continue
        for fn in node.body:
            if isinstance(fn, ast.FunctionDef) and fn.name == method:
                arities = set()
                for r in ast.walk(fn):
                    if isinstance(r, ast.Return) and isinstance(r.value, ast.Tuple):
                        arities.add(len(r.value.elts))
                return arities
    return set()


ENERGIES = [
    ('energies/molecular_crystal.py', 'MolecularCrystal'),
    ('energies/conformer_torsions.py', 'ConformerTorsions'),
]


@pytest.mark.parametrize('path,cls', ENERGIES)
def test_every_energy_returns_the_same_condition_samples_arity(path, cls):
    """The contract itself. If these ever diverge, callers cannot be route-agnostic."""
    arities = _return_arity(ROOT / path, cls, 'condition_samples')
    assert arities == {4}, f'{cls}.condition_samples returns {arities}, expected {{4}}'


def test_no_caller_unpacks_condition_samples_at_the_wrong_arity():
    """THE REGRESSION. One call site unpacked six while every other unpacked four.

    Checked across the whole package rather than at the known site: the bug was not that
    someone wrote six, it was that a signature changed and one caller was left behind, and the
    next signature change will leave a different one behind.
    """
    bad = []
    for py in sorted(ROOT.glob('**/*.py')):
        if any(part in {'.claude', 'tmp', '__pycache__'} for part in py.parts):
            continue
        try:
            tree = ast.parse(py.read_text(encoding='utf-8'))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
                continue
            fn = node.value.func
            if not (isinstance(fn, ast.Attribute) and fn.attr == 'condition_samples'):
                continue
            for target in node.targets:
                if isinstance(target, ast.Tuple) and len(target.elts) != 4:
                    bad.append(f'{py.relative_to(ROOT)}:{node.lineno} unpacks '
                               f'{len(target.elts)}')
    assert not bad, ('condition_samples returns 4 values; these callers disagree:\n  '
                     + '\n  '.join(bad))


def test_bootstrap_z_is_reachable_from_a_conformer_config():
    """The action must PARSE for a conformer protocol, not only for a crystal one.

    Parse-time reachability is the half this test can assert cheaply. Running it needs a
    trained tracker and a live modeller, which `tests/conformer` covers end to end.
    """
    import protocol
    assert 'bootstrap_z' in protocol.ACTIONS
    stage = protocol.Stage({'name': 'tb', 'train_mode': 'fused',
                            'on_enter': ['bootstrap_z'],
                            'loss_coeffs': {'fwd': {'tb': 1.0}}}, 0)
    assert 'bootstrap_z' in [a[0] if isinstance(a, tuple) else str(a).split(':')[0]
                             for a in stage.on_enter]


def test_the_action_list_and_the_parser_agree():
    """An action the parser accepts but nothing dispatches is the same failure one step later."""
    import protocol
    src = inspect.getsource(protocol)
    for action in protocol.ACTIONS:
        assert f"'{action}'" in src, f'{action} is listed but never referenced'


# --------------------------------------------------------------------- the same shape, twice

def test_set_traj_checkpoint_bare_means_on():
    """It used to mean OFF -- so naming the memory panic lever by itself disabled it.

    Every other bare action does the affirmative thing (`freeze_pb` -> 'full'). This one read
    `''` as false, and additionally accepted 'no' / 'n' / 'disabled' as TRUE, because they are
    simply absent from its false-list. The action prints its result, so a typo read as a
    successful line that did the opposite of what was asked.
    """
    import protocol

    class _Args:
        traj_checkpoint = False
        traj_checkpoint_modes = ['fwd']

    class _M:
        args = _Args()
        gfn_model = None
        ema_model = None

    # `stage` is a derived property that walks the whole config; this action only needs it
    # for its log line, so a subclass supplies one rather than standing up a modeller.
    class _SP(protocol.StageProtocol):
        @property
        def stage(self):
            return self._fake_stage

    sp = _SP.__new__(_SP)
    sp.m = _M()
    sp._fake_stage = protocol.Stage({'name': 's', 'train_mode': 'fused',
                                     'loss_coeffs': {'fwd': {'tb': 1.0}}}, 0)

    sp._run_action('set_traj_checkpoint', '', {})
    assert _M.args.traj_checkpoint is True, 'bare set_traj_checkpoint must turn it ON'

    for off in protocol.TRAJ_CKPT_FALSE:
        _M.args.traj_checkpoint = True
        sp._run_action('set_traj_checkpoint', off, {})
        assert _M.args.traj_checkpoint is False, off

    # and an unrecognised argument fails at CONFIG PARSE, not part-way through a run
    with pytest.raises(ValueError, match='unrecognised argument'):
        protocol.Stage({'name': 's', 'train_mode': 'fused',
                        'on_enter': ['set_traj_checkpoint:disabled'],
                        'loss_coeffs': {'fwd': {'tb': 1.0}}}, 0)


def test_set_max_batch_size_validates_at_parse_time():
    """`int(float(arg))` raised mid-transition, with nothing to attribute it to.

    The block that validates `seed_prior_from_anchors` says exactly why: a config typo must
    fail at startup. Two of the actions in that list had been left out of it.
    """
    import protocol
    for bad in ('', 'lots', '0', '-4'):
        with pytest.raises(ValueError, match='set_max_batch_size'):
            protocol.Stage({'name': 's', 'train_mode': 'fused',
                            'on_enter': [f'set_max_batch_size:{bad}'.rstrip(':')],
                            'loss_coeffs': {'fwd': {'tb': 1.0}}}, 0)
    protocol.Stage({'name': 's', 'train_mode': 'fused',
                    'on_enter': ['set_max_batch_size:512'],
                    'loss_coeffs': {'fwd': {'tb': 1.0}}}, 0)


def test_snapshot_prior_refuses_where_prior_model_is_not_the_prior():
    """It deletes best.pt on the premise that `prior_model` IS the prior sampler.

    On a route that draws from a fitted InternalPrior and never reads `prior_model`, the
    premise is false: the action wrote a checkpoint nothing reads, set an attribute nothing
    reads, and deleted best.pt anyway -- silently, with the destructive half the only one
    that landed.
    """
    import protocol

    class _M:
        internal_prior = object()          # this route's backward sampler

    sp = protocol.StageProtocol.__new__(protocol.StageProtocol)
    sp.m = _M()
    with pytest.raises(NotImplementedError, match='never reads `prior_model`'):
        sp._snapshot_prior()
