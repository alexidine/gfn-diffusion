"""Three chart guards on ConformerTorsions, each pinned where its failure would be SILENT.

1. POLYCYCLIC RING SYSTEMS ARE HELD, NOT BANKED. The ring-bank key is (atom count, sorted
   (element, degree) multiset, block DoF count): no topology. Every bank in
   conformer_prior_v2.pt was fitted on a bare monocycle, so a fused, bridged or spiro system
   with the same count and atom types resolved a monocycle's bank and drew a pucker that
   does not close -- median prior energies in the thousands to tens of thousands of kcal/mol,
   against tens when the ring is held. Nothing raised; the rows were just terrible.
2. MMFF IS TYPED ONCE PER CONSTRUCTION. The constructor built ff_single and then `_batch(1)`
   rebuilt the identical force field for the log J probe and e_ref.
3. A SET REFUSES A mol_id-LESS BATCH. condition_samples used to put such rows on condition 0
   -- the first molecule -- on any set, which books every row under the wrong condition.

All at level 'full', the refactor's target tier, unless a test says otherwise.

    python -m pytest -q -p no:cacheprovider tests/conformer/test_conformer_chart_guards.py
"""
import copy
import dataclasses
import pathlib

import numpy as np
import pytest
import torch

from energies.conformer_torsions import ConformerTorsions

pytestmark = pytest.mark.filterwarnings('ignore::UserWarning')

PRIOR_PATH = 'conformer_prior_v2.pt'
N = 64

# Each of these resolved a MONOCYCLE's bank before the guard: the four fused/bridged QM9
# molecules the A/B measured, plus a spiro system, which the key cannot tell from
# cycloheptane either.
POLYCYCLIC = [
    ('bridged, norbornane core', 'CC12CCC(C)(CC1)C2'),        # cycloheptane's key
    ('tricyclic, aziridine-fused', 'CC12CN1C1C(O)C21'),       # piperidine's key
    ('fused 5+3, oxolane', 'CC1OC(C)C2(O)CC12'),              # a 6-ring key
    ('bridged oxa-bicycle', 'CC1(O)C2CCC1O2'),                # tetrahydropyran's key
    ('spiro[2.4]heptane', 'C1CC11CCCC1'),                     # cycloheptane's key
]
# monocycles whose bank genuinely matches: the guard must leave them banked
MONOCYCLIC = [
    ('substituted cyclobutane', 'O=COC1CCC1C=O'),
    ('ethylcyclohexane', 'CCC1CCCCC1'),
    ('cycloheptane', 'C1CCCCCC1'),
]


@pytest.fixture(scope='module')
def prior():
    if not pathlib.Path(PRIOR_PATH).exists():
        pytest.skip('{} missing'.format(PRIOR_PATH))
    return torch.load(PRIOR_PATH, weights_only=False)


@pytest.fixture(scope='module')
def bankless(prior):
    """The same prior with every ring bank removed: the 'hold' arm of the A/B."""
    p = copy.deepcopy(prior)
    p.rings, p.ring_modes = {}, {}
    return p


def _en(smi, level='full', **kw):
    return ConformerTorsions(smiles=smi, level=level, force_field='mmff',
                             log_temperature=0.0, device='cpu', **kw)


def _resolves(prior, key):
    return key in getattr(prior, 'ring_modes', {}) or key in prior.rings


# ------------------------------------------------------ 1: polycyclic systems are held

@pytest.mark.parametrize('name,smi', POLYCYCLIC)
def test_polycyclic_system_is_held_not_banked(prior, bankless, name, smi):
    """A resolved key on a polycycle is REFUSED, and the draw is exactly the hold path.

    The precondition is asserted, not assumed: if a future prior stops resolving this key the
    molecule no longer exercises the guard, and a pass here would then mean nothing.
    """
    en = _en(smi)
    en.ring_blocks(prior)
    info = en.ring_block_info
    assert len(info) == 1, (name, info)
    assert _resolves(prior, info[0]['key']), (
        '{}: the prior no longer resolves this ring key, so the molecule cannot exercise the '
        'polycyclic refusal -- pick another'.format(name))

    # THE SAME DRAW AS A PRIOR WITH NO BANKS AT ALL -- bitwise, from the same seed. Asserted
    # on the draw BEFORE the record, because a record can be written while the sampler still
    # takes the bank branch.
    x_b, st_b = en.sample_prior_states(prior, N, np.random.default_rng(0), report=False)
    # the record THIS draw wrote -- the bankless draw below re-runs ring_blocks and replaces it
    rec = dict(en.ring_block_info[0])
    _, st_h = en.sample_prior_states(bankless, N, np.random.default_rng(0), report=False)
    assert st_b['n_ring_banked'] == 0 and st_b['n_ring_thermal'] == 1, (
        '{}: banked {} held {}'.format(name, st_b['n_ring_banked'], st_b['n_ring_thermal']))
    assert np.array_equal(st_b['dof'], st_h['dof']), (
        '{}: the refused-bank draw differs from the no-bank draw'.format(name))

    # and the consequence the guard exists for: closed rings and a sane prior energy
    from energies.conformer_data import bake_energies
    e = bake_energies(en, x_b)
    assert st_b['closure_sigma'] < 3.0, (name, st_b['closure_sigma'])
    assert float(e.median()) < 500.0, (
        '{}: median prior energy {:.0f} kcal/mol at level full -- the ring is not being '
        'held'.format(name, float(e.median())))

    # the record says WHY: a key resolved and was refused, not "no bank for this ring"
    assert rec['ring_class'] == 'held_unsupported', rec
    assert rec.get('n_cycles', 0) >= 2, rec
    assert rec.get('bank_refused') == 'polycyclic', rec


@pytest.mark.parametrize('name,smi', MONOCYCLIC)
def test_monocycle_bank_still_applies(prior, name, smi):
    """The guard is a topology gate, not a blanket hold: a genuine monocycle stays banked."""
    en = _en(smi)
    en.ring_blocks(prior)
    (rec,) = en.ring_block_info
    assert rec['n_cycles'] == 1, rec
    assert rec['bank_refused'] is None, rec
    assert rec['ring_class'] in ('banked_modes', 'banked_rows'), rec
    _, st = en.sample_prior_states(prior, N, np.random.default_rng(0), report=False)
    assert st['n_ring_banked'] == 1, st


def test_aromatic_polycycle_is_held_by_design_not_refused(prior):
    """Aromaticity is decided FIRST: naphthalene is held_aromatic, not a refused bank.

    Otherwise the refusal would relabel a held-by-design ring as a gap in the fit.
    """
    en = _en('c1ccc2ccccc2c1')
    en.ring_blocks(prior)
    (rec,) = en.ring_block_info
    assert rec['n_cycles'] == 2 and rec['ring_class'] == 'held_aromatic', rec
    assert rec['bank_refused'] is None, rec


@pytest.mark.parametrize('smi', ['CC12CCC(C)(CC1)C2', 'CC12CN1C1C(O)C21', 'C1CC11CCCC1',
                                 'C1CCC(CO1)c1ccccc1', 'CCC1CCCCC1', 'C1CCC2CCCCC2C1'])
def test_cycle_count_agrees_with_rdkit(prior, smi):
    """The per-system count, summed, is RDKit's SSSR size -- an independent implementation.

    The count is E - V + 1 over each system's bonds, read in the numbering of the ring-system
    ids. A numbering slip would still give a number, so it is checked against a count that
    shares none of that plumbing.
    """
    from rdkit import Chem
    en = _en(smi)
    en.ring_blocks(prior)
    got = sum(r['n_cycles'] for r in en.ring_block_info)
    sssr = Chem.GetSSSR(Chem.Mol(en.mol))
    want = sssr if isinstance(sssr, int) else len(sssr)
    assert got == want, (smi, got, want, en.ring_block_info)


# ---------------------------------------------------------- 2: MMFF typed once

def _count_calls(monkeypatch, name):
    import mxtaltools.conformers.energy as mce
    real = getattr(mce, name)
    calls = []

    def counted(*a, **kw):
        calls.append(1)
        return real(*a, **kw)

    monkeypatch.setattr(mce, name, counted)
    return calls


@pytest.mark.parametrize('ff_choice,builder', [('mmff', 'ff_from_mmff'),
                                               ('reference', 'ff_from_reference')])
def test_force_field_is_built_once_per_construction(monkeypatch, ff_choice, builder):
    """One force-field build per ConformerTorsions, and the counter is shown to be LIVE.

    `_make_ff` imports the builder at call time, so patching the module attribute intercepts
    it. A count of 1 would also be what a patch that intercepted nothing reports on some other
    path, so the test then forces a cache miss and requires the counter to move.
    """
    calls = _count_calls(monkeypatch, builder)
    en = ConformerTorsions(smiles='CCC1CCCCC1', level='full', force_field=ff_choice,
                           device='cpu')
    assert len(calls) == 1, '{} called {} times in one construction'.format(builder,
                                                                             len(calls))
    en._tree_cache.clear()
    en._ff_cache.clear()
    en._batch(1)
    assert len(calls) == 2, 'the counter did not see a cache-miss rebuild; it is not live'


@pytest.mark.parametrize('level', ['dihedral', 'full'])
def test_seeded_batch1_entry_equals_a_cold_rebuild(level):
    """The seeded entry is the force field `_batch(1)` would have built, field for field.

    And the constants the constructor derived from it -- e_ref, and log_jacobian_const where
    the level has one -- reproduce exactly from a cold cache.
    """
    en = _en('CCC1CCCCC1', level=level)
    seeded = en._ff_cache[1]
    assert seeded is en.ff_single
    e_ref, ljc = en.e_ref, en.log_jacobian_const
    en._tree_cache.clear()
    en._ff_cache.clear()
    tree, cold = en._batch(1)
    assert cold is not seeded
    for f in dataclasses.fields(seeded):
        a, b = getattr(seeded, f.name), getattr(cold, f.name)
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b), f.name
        else:
            assert a == b, f.name
    zero = torch.zeros(1, en.data_ndim, dtype=en.dtype)
    assert en.energy(zero, None, torch.tensor(en.log_temperature)).item() == e_ref
    if ljc is not None:
        r, th, ph = en.dof_from_state(zero)
        assert float(en._log_jac(tree, r, th, ph, 1).item()) == ljc
    assert (ljc is None) == (level == 'full')


# ----------------------------------------------- 3: a set refuses a mol_id-less batch

def _two_row_batch(en):
    from energies.conformer_data import collate_conditions, condition_from_energy
    return collate_conditions([condition_from_energy(en, identifier='a'),
                               condition_from_energy(en, identifier='b')])


def test_single_molecule_route_still_defaults_to_condition_zero():
    """Library size 1 is one condition, so a missing mol_id there is not an error."""
    en = _en('CCCCO')
    assert en.condition_library_size == 1
    _, _, _, cid = en.condition_samples(_two_row_batch(en))
    assert cid.tolist() == [0, 0]


def test_set_refuses_a_batch_without_mol_id():
    """On a set the zeros fallback would book every row under the first molecule."""
    en = _en('CCCCO', temperature_conditioning=True)
    en.set_n_molecules(2)
    batch = _two_row_batch(en)
    rng = torch.get_rng_state()
    with pytest.raises(RuntimeError, match='mol_id'):
        en.condition_samples(batch)
    # refused BEFORE the temperature draw, so the refusal consumed no randomness
    assert torch.equal(rng, torch.get_rng_state())
    assert getattr(batch, 'condition_id', None) is None

    # with mol_id the same batch conditions normally, one condition per row
    batch.add_graph_attr(torch.tensor([1, 0], dtype=torch.long), 'mol_id')
    _, _, _, cid = en.condition_samples(batch)
    assert cid.tolist() == [1, 0]


def test_multi_molecule_set_refuses_a_batch_without_mol_id():
    """The real consumer: MultiConformerTorsions inherits condition_samples unchanged.

    The two members differ in atom count, so this is a CARRIER set -- the batch is built
    from a plain single-molecule energy because condition_samples reads only the batch's
    rows and attributes, never the chart.
    """
    from energies.multi_conformer import MultiConformerTorsions
    en = MultiConformerTorsions(['CCCCO', 'CCCCN'], level='full', force_field='mmff',
                                device='cpu')
    en.set_n_molecules(en.n_charts)
    assert en.condition_library_size == 2
    with pytest.raises(RuntimeError, match='mol_id'):
        en.condition_samples(_two_row_batch(_en('CCCCO')))
