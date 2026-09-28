"""A full resume REFUSES another condition set at startup; a weights-only load allows it.

WHAT WAS WRONG. mol_id is an identifier's position in the sorted condition set
(train.py::Modeller.init_identifiers), and the checkpoint did not record which identifiers those
were. In the G0 gate (2026-09-27) a conditions file with one member swapped for another of the
same size (CCO -> COC) loaded without complaint and failed at the first backward draw, in
MultiConformerTorsions._resolve_rows, after the restored tracker and buffers had been keyed to
the shifted mol_ids; an added member was refused only by init_condition_log_z's tracker size
check, after the buffer sidecar had been restored.

WHAT IS PINNED, through the real ConformerModeller.init_gfn, Modeller.init_gfn and
Checkpointer.save / load_full / load_weights_only, on test_set_policy_resume.py's stub modeller:

  * the stamp: gfn_config['conformer']['condition_set'] holds the identifiers in mol_id order and
    one signature per member, on the set head and on a flat policy over a carrier, and separately
    built energies of one set sign it identically;
  * a full resume onto a same-size substitute, an added member, a permuted stamp or a changed
    member is refused inside load_full BEFORE the modeller state, the buffer sidecar or the
    tracker are restored (spied), naming the identifiers and both ways on -- and a requeue
    (continue_from_checkpoint) is a full load whatever load_weights_only says;
  * a reordered conditions FILE cannot move a mol_id, because registration sorts: it resumes;
  * the identical set resumes and says so on one line;
  * a weights-only load onto another set is allowed, named on one line, and re-stamped;
  * a checkpoint written before the stamp warns once on a full resume and loads;
  * init_identifiers refuses a registry that is not the stamped list.

    CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/conformer/test_condition_set_identity.py
"""
import os
from types import SimpleNamespace

import pytest
import torch

from conformer_modeller import ConformerModeller
from energies.multi_conformer import MultiConformerTorsions
from energy_sampling.buffer import ConditionLogZTracker
from test_set_policy_resume import KW, SMIS, _args, _modeller

# small models and five small molecule sets on CPU, nothing off the data drive
pytestmark = pytest.mark.fast

#: the three restores load_full makes after it builds the model: modeller state, then the
#: buffer sidecar (load_buffers_for -> restore_buffers). The tracker is read by hasattr.
_RESTORES = ('set_state_dict', 'load_buffers_for', 'restore_buffers')


@pytest.fixture(scope='module', autouse=True)
def _float64():
    """float64 for the module, as test_set_policy_resume.py, RESTORED after: a module-scope
    default leaks into every test collected after this one."""
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _set(smis, idents=None):
    return MultiConformerTorsions(smis, identifiers=idents or smis, **KW)


@pytest.fixture(scope='module')
def carrier():
    return _set(SMIS)                      # C, CO, N: full/mmff, carrier K = 12 (5|4|3)


@pytest.fixture(scope='module')
def swapped():
    """N swapped for C=O, both 4 atoms at 3|2|1: K, block_width and the periodic columns are the
    checkpoint's, so only the condition set differs -- G0's CCO -> COC in miniature."""
    return _set(['C', 'CO', 'C=O'])


@pytest.fixture(scope='module')
def plus_one():
    return _set(SMIS + ['C=O'])            # one member more, inside the same widths


@pytest.fixture(scope='module')
def reordered():
    return _set(SMIS[::-1])                # the same members, listed N, CO, C


@pytest.fixture(scope='module')
def changed():
    """'N' names formaldehyde here: the identifiers are the checkpoint's, one member is not."""
    return _set(['C', 'CO', 'C=O'], ['C', 'CO', 'N'])


@pytest.fixture(scope='module')
def saved(carrier, tmp_path_factory):
    """Modeller A over SMIS, built fresh through the real init_gfn, carrying a 3-row tracker at
    step 7 so a restore of either is visible; saved as 'probe' with its buffer sidecar, and as
    'running' for the requeue path."""
    ckdir = tmp_path_factory.mktemp('ck_set')
    a = _modeller(carrier, _args(ckdir))
    a.init_gfn()
    a.condition_log_z = ConditionLogZTracker(library_size=3)
    a.step_ind = 7
    a.checkpointer.save('probe', with_buffers=True)
    a.checkpointer.save('running')
    return a, ckdir, a.checkpointer.path_for('probe')


def _loader(energy, ckdir, path=None, run_name='reload', **kw):
    """test_set_policy_resume's `_load` split in two, so the restores load_full makes are
    recorded on `.restores` and the modeller survives a refused init_gfn for inspection."""
    if path is not None:
        kw['checkpoint_name'] = os.path.basename(str(path))
    b = _modeller(energy, _args(ckdir, **kw), run_name=run_name)
    b.restores = []
    for name in _RESTORES:
        real = getattr(b.checkpointer, name)

        def spy(*a, _name=name, _real=real, **k):
            b.restores.append(_name)
            return _real(*a, **k)
        setattr(b.checkpointer, name, spy)
    return b


def _refused(b):
    """init_gfn raises the condition-set refusal with nothing restored; returns the message."""
    with pytest.raises(ValueError, match='condition set') as e:
        b.init_gfn()
    assert b.restores == [], f'restored {b.restores} before refusing'
    assert not hasattr(b, 'condition_log_z') and b.step_ind == 0
    msg = str(e.value)
    assert 'fresh' in msg and 'load_weights_only: true' in msg, 'both ways on are named'
    return msg


def _tampered(path, dst, edit):
    """A copy of the checkpoint at `path` whose stored conformer block `edit` rewrites."""
    ck = torch.load(path, weights_only=False)
    ck['gfn_config'] = dict(ck['gfn_config'])
    ck['gfn_config']['conformer'] = edit(dict(ck['gfn_config']['conformer']))
    torch.save(ck, dst)
    return dst


# ------------------------------------------------------------------ the stamp


def test_the_stamp_names_the_sorted_set_on_the_set_head_and_on_a_flat_carrier(saved, carrier,
                                                                               tmp_path):
    a, _, _ = saved
    cs = a.gfn_config['conformer']['condition_set']
    assert cs['identifiers'] == sorted(SMIS) == ['C', 'CO', 'N']
    assert len(cs['signatures']) == 3 and all(len(s) == 16 for s in cs['signatures'])
    assert cs['signature_recipe'] == ConformerModeller._MEMBER_SIGNATURE
    f = _modeller(carrier, _args(tmp_path, model__policy_kind='flat'), run_name='flat')
    f.init_gfn()
    assert f.gfn_config['conformer']['policy_kind'] == 'flat'
    assert f.gfn_config['conformer']['condition_set'] == cs


def test_the_signature_is_a_function_of_the_member(carrier, reordered, changed):
    """Separately built energies of one set sign it identically -- a resume rebuilds them -- and
    the digest moves with the member, not with the identifier."""
    def identity(en):
        return _modeller(en, _args('.'))._condition_set_identity()

    assert list(reordered._members) != sorted(reordered._members), \
        'separator: this energy lists its members in another order'
    assert identity(reordered) == identity(carrier)
    mine, other = identity(carrier), identity(changed)
    assert other['identifiers'] == mine['identifiers']
    assert [i for i, x, y in zip(mine['identifiers'], mine['signatures'], other['signatures'])
            if x != y] == ['N']


# ------------------------------------------------------------------ full resume: refused


def test_a_same_size_substitution_is_refused_before_anything_is_restored(saved, carrier,
                                                                         swapped):
    _, ckdir, path = saved
    assert (swapped.data_ndim, swapped.carrier.block_width) == \
        (carrier.data_ndim, carrier.carrier.block_width), \
        'separator: the layout is the checkpoint\'s, so only the condition set can refuse this'
    msg = _refused(_loader(swapped, ckdir, path))
    assert "only in the checkpoint ['N']" in msg and "only in this run ['C=O']" in msg
    assert "mol_id moved ['CO' 1 -> 2]" in msg


def test_an_added_member_is_refused_naming_it(saved, plus_one):
    _, ckdir, path = saved
    msg = _refused(_loader(plus_one, ckdir, path))
    assert "only in this run ['C=O']" in msg and 'only in the checkpoint' not in msg


def test_a_changed_member_under_the_same_identifier_is_refused(saved, changed):
    _, ckdir, path = saved
    msg = _refused(_loader(changed, ckdir, path))
    assert "another member (SMILES, block codes or placement-order atoms) ['N']" in msg
    assert 'only in' not in msg and 'mol_id moved' not in msg


def test_a_stamp_in_another_order_is_refused(saved, carrier, tmp_path):
    """The comparison is ORDER-sensitive. A reordered set cannot come from the conditions file
    (see the reorder test below); this is what a registration that did not sort would have
    written, and it is refused naming the moved identifiers."""
    _, _, path = saved

    def permute(block):
        cs = dict(block['condition_set'])
        cs['identifiers'] = [cs['identifiers'][k] for k in (0, 2, 1)]     # C, N, CO
        cs['signatures'] = [cs['signatures'][k] for k in (0, 2, 1)]
        return dict(block, condition_set=cs)

    bad = _tampered(path, tmp_path / 'perm_conformer-test_probe.pt', permute)
    msg = _refused(_loader(carrier, tmp_path, bad))
    assert "'CO' 2 -> 1" in msg and "'N' 1 -> 2" in msg
    assert 'only in' not in msg and 'another member' not in msg


def test_a_requeue_is_a_full_load_whatever_load_weights_only_says(saved, swapped):
    """continue_from_checkpoint takes train.py's load_full branch; load_weights_only is read only
    beside checkpoint_name, so a stray true must not turn a requeue into the allowed path."""
    a, ckdir, _ = saved
    _refused(_loader(swapped, ckdir, run_name=a.run_name, continue_from_checkpoint=True,
                     load_weights_only=True))


# ------------------------------------------------------------------ allowed


def test_reordering_the_conditions_file_cannot_move_a_mol_id(saved, reordered):
    """Registration SORTS the identifiers, so a file listing the same members in another order
    registers the same mol_ids: a reordered set cannot arise from the conditions file."""
    a, ckdir, path = saved
    b = _loader(reordered, ckdir, path)
    b.init_gfn()
    assert b.step_ind == 7
    assert b.gfn_config['conformer']['condition_set'] == a.gfn_config['conformer']['condition_set']


def test_the_identical_set_resumes_and_says_so(saved, carrier, capsys):
    a, ckdir, path = saved
    capsys.readouterr()
    b = _loader(carrier, ckdir, path)
    b.init_gfn()
    out = capsys.readouterr().out
    assert ("condition set: 3 identifiers in mol_id order and their member signatures match "
            "the checkpoint's") in out
    # separator: the spies see a real load, in load_full's order, after the check passed
    assert b.restores[:2] == ['set_state_dict', 'load_buffers_for']
    assert b.step_ind == 7 and int(b.condition_log_z.library_size) == 3
    assert b.gfn_config['conformer'] == a.gfn_config['conformer']


def test_a_weights_only_load_onto_another_set_is_allowed_and_named_on_one_line(saved, swapped,
                                                                             capsys):
    _, ckdir, path = saved
    capsys.readouterr()
    w = _loader(swapped, ckdir, path, load_weights_only=True)
    w.init_gfn()
    lines = [ln for ln in capsys.readouterr().out.splitlines()
             if ln.startswith('condition set:')]
    assert len(lines) == 1 and 'weights-only' in lines[0], lines
    assert "['N']" in lines[0] and "['C=O']" in lines[0]
    assert w.weights_only_loaded is True and w.restores == []
    assert not hasattr(w, 'condition_log_z'), 'a weights-only load restores no tracker'
    assert w.gfn_config['conformer']['condition_set']['identifiers'] == ['C', 'C=O', 'CO'], \
        "re-stamped: this run's checkpoints name the set its tracker and buffers are keyed to"


def test_a_checkpoint_written_before_the_set_stamp_warns_once_and_loads(saved, carrier, swapped,
                                                                      tmp_path, capsys):
    """The block_width convention: an unstamped file is warned about, not refused -- even when
    the set differs at the same size, which is the gap the warning names."""
    _, _, path = saved
    old = _tampered(path, tmp_path / 'old_conformer-test_probe.pt',
                    lambda block: {k: v for k, v in block.items() if k != 'condition_set'})
    for energy, want in ((carrier, ['C', 'CO', 'N']), (swapped, ['C', 'C=O', 'CO'])):
        capsys.readouterr()
        b = _loader(energy, tmp_path, old)
        b.init_gfn()
        out = capsys.readouterr().out
        assert out.count('checkpoint carries no condition-set stamp') == 1, out
        assert b.step_ind == 7, 'a full load, warned and not refused'
        assert b.gfn_config['conformer']['condition_set']['identifiers'] == want, \
            'this leg stamps its own set'


# ------------------------------------------------------------------ the registry


def _dataset(idents):
    """What Modeller.init_identifiers reads off a dataset: the batch's identifiers, its device
    and add_graph_attr."""
    return SimpleNamespace(batch=SimpleNamespace(identifier=list(idents), device='cpu',
                                                 add_graph_attr=lambda value, name: None))


def test_init_identifiers_refuses_a_registry_that_is_not_the_stamp(carrier, tmp_path):
    """The stamp is computed from the members before any dataset loads; the registry is the
    sorted union of the loaded datasets' identifiers. A prior naming an identifier with no
    member breaks the equality, and is refused at init rather than at its first draw."""
    m = _modeller(carrier, _args(tmp_path))
    m.init_gfn()
    m.mol_dataset, m.test_mol_dataset = _dataset(SMIS), None
    m.prior_dataset = _dataset(SMIS + ['CCO'])
    with pytest.raises(ValueError, match=r"only in the registry \['CCO'\]"):
        m.init_identifiers()
    m.prior_dataset = _dataset(SMIS[::-1])
    m.init_identifiers()
    assert m.identifier_registry == {'C': 0, 'CO': 1, 'N': 2}
    assert m.gfn_config['conformer']['condition_set']['identifiers'] == ['C', 'CO', 'N']
