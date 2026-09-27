"""Write the conformer condition file, and optionally the graph-form prior beside it.

Two artifacts, the same split train.py's data layer already makes (see
``energies/conformer_data.py`` for the formats and why):

  ``--out``        the condition set -> ``molecules_path``. One graph per molecule,
                   carrying the internal-coordinate tree, the reference conformer and the
                   state-column map. No state.
  ``--prior-out``  the prior -> ``prior_path``. Every molecule replicated once per drawn
                   state, with the state and its raw energy baked in, under
                   ``equalized_prior``.

This is the AD HOC builder: a SMILES list in, one file out. Production condition sets --
QM9 subsets with a held-out split, stereoisomer enumeration, a rejection table and a
manifest -- come from ``build_conformer_set.py``, which reuses the per-molecule step here
(``build_member``) so the two cannot disagree about what a member is.

    # one molecule, conditions + prior, checked against the energy
    python build_conformer_conditions.py --smiles CCCCO --prior-out conformer_prior_graphs.pt \\
        --internal-prior conformer_prior_v2.pt

    # a mixed-k set in the carrier layout, energy kwargs taken from the run's config
    python build_conformer_conditions.py --smiles C CO N CCO C=O CC --carrier \\
        --config configs/conformer_mk.yaml --out conformer_conditions.pt

A plain file (no ``--carrier``) is ONE k, because the GFN's state dimension is fixed at
construction; molecules of another k are skipped and named. ``--carrier`` writes a mixed-k
set in the width-K carrier layout instead (energies/conformer_carrier.py).

EVERY STEP OF ``build_member`` IS GUARDED AND EVERY REFUSAL HAS A REASON CODE
(``REASON_CODES``): an exception anywhere in it becomes ``MemberRefused``, classified by its
text, 'other' with the message kept when no pattern matches. Before this, only the
``ConformerTorsions`` constructor sat inside the try: a ring-closure encoding failure in
``condition_from_energy`` (0.8% of QM9) aborted the build after every earlier molecule was
processed, and one nitrile made ``CarrierLayout`` raise for the WHOLE set (it has since
placed transverse columns, and admits nitriles). A refusal now drops that molecule, names
it, and the file is written from the rest. What is NOT a
per-molecule refusal: the set-level carrier check after every member is padded into the
shared layout (``main`` here, ``build_conformer_set.main``). Each member already passed the
same check in a one-member layout, so a failure there is a layout defect, and it aborts.

STEREO IS PART OF A CONDITION'S IDENTITY. A molecule with a stereo element (a tetrahedral
centre, ring cis/trans, a C=C or C=N bond) must arrive with every such element specified,
and the embedded reference is re-perceived from 3D and must be the same stereoisomer. An
untagged chiral SMILES is refused (``stereo_unspecified``): ETKDG would pick an isomer by
seed and the identifier would not say which. Tetrahedral N is exempt -- see ``stereo_mol``.
"Same stereoisomer" is ``stereo_identity``, never string equality of canonical SMILES; see
there for why.
"""

import argparse
import contextlib
import functools
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from energies.conformer_data import (attach_states, bake_energies, check_state_convention,
                                     collate_conditions, condition_from_energy,
                                     save_condition_file, save_prior_file)
from energies.conformer_torsions import ConformerTorsions
from energies.dof_features import free_dof_atom_index
from models import encoder_cache
from paths import artifact

#: every code a per-molecule refusal is recorded under, with what it means. `build_member`
#: raises `MemberRefused(code, message)`; build_conformer_set.py adds its set-level codes.
REASON_CODES = {
    'lt4_atoms': 'fewer than 4 atoms: no internal parameterisation exists',
    'embed_failed': "the member constructor's seeded ETKDG could not embed the SMILES, and no "
                    'untagged embedding of the constitution realised this stereoisomer. '
                    'NOT proof the isomer cannot exist: ETKDG with chirality constraints '
                    'also fails on isomers that strained cages do realise',
    'embed_failed_realisable': "the member constructor's seeded ETKDG could not embed this "
                               'stereo-tagged SMILES, but an UNTAGGED embedding of the same '
                               'constitution realises exactly this stereoisomer: an '
                               'embedding limitation on a strained cage, not physics',
    'mmff_typing': 'the force field could not type or parameterise the molecule',
    'graph_broken': "the geometry-inferred bond graph the chart is built on differs from "
                    "RDKit's bond graph",
    'incomplete_chart_full': "a constrained chart at level 'full': a row is held at a linear "
                             "centre that neither a transverse pair nor a dummy frame carries",
    'theta_box': 'the theta box reference +/- delta_theta_max leaves (0, pi)',
    'r_box': 'the r box floor reaches r_floor',
    'transverse_box': 'a transverse bend can reach rho >= pi inside the box (rho >= pi/2 '
                      'for a bend that anchors a dummy frame)',
    'no_free_dof': 'no free coordinate at this level',
    'chart_defect': "torsion tier only: energy.mask and _M disagree on the column count",
    'k_mismatch': 'state width differs from the file k (plain, non-carrier file)',
    # NO `transverse_no_carrier_block`: CarrierLayout places a transverse column in the theta
    # region and raises only on a block code with no region, a chart/layout defect rather
    # than a molecule's property, which lands in 'other' with its message
    'closure_encoding': 'a ring-closure bond with no free endpoint for the per-atom '
                        'closure encoding',
    'convention_check': 'the condition graph and the energy build different geometry',
    'stereo_unspecified': 'the SMILES leaves stereo open: it enumerates to more than one '
                          'stereoisomer (enumerate_stereoisomers)',
    'stereo_n_tagged': 'the SMILES tags a tetrahedral N, which 3D perception cannot verify',
    'stereo_verify_failed': 'the embedded reference, re-perceived from 3D, is a different '
                            'stereoisomer (stereo_identity) from the one requested',
    'encoder_alignment': 'the encoder atom order does not align with the tree',
    'wholly_linear': 'every atom on one line: 3N-5 internal DoF, so no 3N-6 chart exists',
    'cumulated': 'a collinear frame through a cumulated sp centre (allene, cumulene), held '
                 'because freeing it frees a twist the force field does not restrain',
    'stereo_unsupported': 'a stereo element the lock does not enforce is tagged (tetrahedral '
                          'N, an allene or atropisomer axis)',
    'stereo_lock_in_band': "an element's indicator sits too near zero at the reference, or "
                           "the lock fired on a thermal sample of the correct isomer",
    'stereo_torsion_double_bond': 'at torsion a rotatable column turns a locked double bond',
    'other': 'anything else; the message is recorded',
}

# ORDERED: the first pattern found in the exception text wins. These are the refusal
# messages ConformerTorsions, the MXtalTools tree/force-field builders,
# condition_from_energy and encoder_cache raise today (a ChartRefused's own code, through
# `_CHART_CODES`, overrides any pattern). A new refusal elsewhere lands in 'other' WITH its
# message, rather than being guessed into a neighbouring code.
_FAILURE_PATTERNS = (
    ('lt4_atoms', 'need at least 4 atoms'),
    ('embed_failed', 'could not embed'),
    ('mmff_typing', 'could not MMFF-type'),
    ('mmff_typing', 'MMFF has no'),
    ('mmff_typing', 'untyped'),
    ('graph_broken', 'perceived graph is disconnected'),
    ('graph_broken', 'bond perception found no bonds'),
    ('incomplete_chart_full', 'does not have a complete chart'),
    ('theta_box', 'puts the theta box'),
    ('r_box', 'puts the box floor'),
    ('transverse_box', 'transverse bend reach'),
    ('transverse_box', 'anchors a dummy frame reach'),
    ('no_free_dof', 'no free degrees of freedom'),
    ('no_free_dof', 'has no rotatable bonds'),
    ('closure_encoding', 'ring-closure bond'),
    ('encoder_alignment', 'ATOM ORDER MISMATCH'),
    ('encoder_alignment', 'spec.perm has'),
)

#: ConformerTorsions.ChartRefused.code -> this table's code
_CHART_CODES = {'incomplete_chart': 'incomplete_chart_full', 'wholly_linear': 'wholly_linear',
                'cumulated': 'cumulated', 'stereo_unspecified': 'stereo_unspecified',
                'stereo_unsupported': 'stereo_unsupported',
                'stereo_verify_failed': 'stereo_verify_failed',
                'stereo_lock_in_band': 'stereo_lock_in_band',
                'stereo_torsion_double_bond': 'stereo_torsion_double_bond'}

#: untagged embeddings per constitution in the realisability probe (``untagged_realisations``)
PROBE_SEEDS = 4

#: energy_config keys that describe the RUN, not the chart or the energy, and so are not
#: forwarded to the member a file is built against (the modeller sets them itself)
_RUN_SURFACE_KEYS = ('smiles', 'device', 'dtype', 'temperature_conditioning',
                     'embedding_conditioning', 'embedding_conditioning_dim',
                     'log_temperature_range')


class MemberRefused(Exception):
    """One molecule could not become a condition. ``code`` is a key of REASON_CODES."""

    def __init__(self, code: str, message: str):
        super().__init__(f'{code}: {message}')
        self.code, self.message = code, message


def classify_failure(exc: BaseException) -> str:
    text = f'{type(exc).__name__}: {exc}'
    for code, pattern in _FAILURE_PATTERNS:
        if pattern in text:
            return code
    return 'other'


def energy_kwargs_from_config(path) -> tuple:
    """``(kwargs, energy_config)``: the ConformerTorsions kwargs a run's config implies.

    FILTERED BY THE SAME RULE THE RUN USES: ``ConformerModeller.init_energy_function`` passes
    every energy_config key the ``ConformerTorsions`` signature accepts. A file built from
    these kwargs therefore carries the chart the run's members will build -- the scales, the
    floors, the force field, the seed, the energy clip -- rather than the builder's own
    defaults, which is how a file and a run silently disagreed before (VEC-9, DATA-17).
    Keys naming the run surface are dropped (``_RUN_SURFACE_KEYS``). ``energy_config`` is
    returned whole for the non-chart keys a builder also reads (internal_prior_path,
    prior_relax_steps).
    """
    import yaml

    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    ec = dict((cfg or {}).get('energy_config') or {})
    accepted = set(inspect.signature(ConformerTorsions.__init__).parameters) - {'self'}
    kw = {k: v for k, v in ec.items() if k in accepted and k not in _RUN_SURFACE_KEYS}
    return kw, ec


# ------------------------------------------------------------------ stereochemistry


@contextlib.contextmanager
def legacy_stereo():
    """Pin RDKit's LEGACY stereo perception for the duration.

    Every stereo decision below -- which elements exist, whether each is specified, what a
    3D reference re-perceives as -- reads process-wide RDKit state. The non-legacy
    perception flags spurious centres (a gem-dimethyl carbon, a symmetric spiro atom) on
    about 10% of QM9, and RDKit intends to change the default. Pinning it makes the
    identifiers a function of the molecule, not of whoever set the flag last.
    """
    from rdkit import Chem

    was = Chem.GetUseLegacyStereoPerception()
    Chem.SetUseLegacyStereoPerception(True)
    try:
        yield
    finally:
        Chem.SetUseLegacyStereoPerception(was)


def stereo_mol(smiles: str):
    """The molecule on which stereo is enumerated and judged: implicit H, EXCEPT on =NH.

    NEITHER HYDROGEN CONVENTION SEES EVERY ELEMENT. With all H explicit, RDKit misses ring
    cis/trans (1,4-cyclohexanediol and 1,3-dimethylcyclobutane enumerate to ONE isomer);
    with all H implicit it misses imine E/Z, because the =NH end has no heavy controlling
    atom. So hydrogens are made explicit only on a double-bonded atom whose one other
    substituent is a lone H -- the =NH case -- which catches both.

    Tetrahedral N is not a stereo element here. RDKit's 3D perception does not recover N
    configuration, so an N-tagged isomer could not be verified, and ordinary amines invert.
    WORKING ASSUMPTION (scope: this builder; revisit if aziridine or bridgehead-N molecules
    show a prior-coverage defect): N invertomers stay together inside one condition. Where
    inversion is hindered (an aziridine N, a ring-fused N) the two invertomers are separate
    basins, one condition spans both, and the reference sits in one of them. A bridgehead N
    in a cage has no second invertomer; its configuration follows from the C centres, which
    keep their tags.
    """
    from rdkit import Chem

    m = Chem.MolFromSmiles(smiles)
    if m is None:
        raise ValueError(f'unparseable SMILES: {smiles}')
    on = [a.GetIdx() for a in m.GetAtoms()
          if a.GetDegree() == 1 and a.GetTotalNumHs() == 1
          and any(b.GetBondType() == Chem.BondType.DOUBLE for b in a.GetBonds())]
    return Chem.AddHs(m, onlyOnAtoms=on) if on else m


def _strip_n_tags(m):
    from rdkit import Chem

    for a in m.GetAtoms():
        if a.GetAtomicNum() == 7 and a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
            a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    return m


def _parse(smiles: str):
    from rdkit import Chem

    m = Chem.MolFromSmiles(smiles)
    if m is None:
        raise ValueError(f'unparseable SMILES: {smiles}')
    return m


def stereo_identity(mol) -> tuple:
    """WHICH STEREOISOMER ``mol`` is, read off its atoms and bonds without the SMILES writer.

    ``(fixed-H InChI, CIP signature)``; two molecules are one stereoisomer only when BOTH
    agree. Call it under ``legacy_stereo`` (``mol``'s tags were assigned under it).

    RDKIT'S CANONICAL SMILES IS NOT A STEREO IDENTITY on some cages and spiro systems, so
    no identifier comparison goes through it. Measured over 3,000 random QM9 molecules at
    index >= 25,364 (RDKit 2025.03.5; ``stereoisomer_classes``):

      * two strings, each a fixed point of the writer, name ONE isomer: 8 of the 17,685
        strings enumeration writes, in 7 molecules (QM9 40520 -- equal InChI, MMFF
        references 2e-5 A apart -- 55768, 67063, 122753, and three 1,3-disubstituted
        bicyclo[1.1.1]pentanes). As strings they became two conditions of one molecule;
      * re-writing a name CHANGES the isomer for 10 of the 17,677 names (QM9 32293, 67063):
        ``C1[C@@H]2[C@H]3C[C@H]4[C@@H]2[C@@H]1[C@@H]34`` re-writes to a SMILES whose InChI
        t-layer differs, so a reference compared against a re-canonicalised identifier
        failed a check it passes.

    The two readings, and why both. InChI with the fixed-H layer: standard InChI treats
    amidine/guanidine H as mobile and drops the C=N stereo -- it would merge 134 pairs of
    those 17,677 distinct isomers; fixed-H still merges 3, on amidinium zwitterions. The CIP
    labels from ``rdCIPLabeler`` (the new labeller, not legacy perception), keyed by
    stereo-free canonical atom rank, separate those 3; a label multiset alone is not a
    complete identity (positional patterns round a ring), which InChI is. The labeller is
    not independent of atom order on every cage (see ``stereoisomer_classes``): between two
    orders it can split one isomer into two identities, while merging two isomers would
    need InChI to fail with it. The member check compares a parsed string with the member
    built from that string, which share one atom order.

    A TAG THE CIP LABELLER DOES NOT LABEL IS DROPPED FIRST: the labeller is the arbiter of
    what is a stereo element. Legacy perception and InChI both keep parities on the two
    bridgeheads of a bicyclo[1.1.1]pentane (``O[C@]12C[C@@](O)(C1)C2`` and its flip get
    different InChI), which have none, and ``AssignStereochemistryFrom3D`` puts them there,
    so without this every such reference would fail its check. The labeller does label ring
    cis/trans (r/s on 1,4-cyclohexanediol) and pseudo-asymmetric centres. Stale ``_CIPCode``
    props from legacy perception are cleared before it runs. Tetrahedral N tags are stripped
    (``stereo_mol`` says why) and conformers dropped, so InChI reads the tags rather than
    re-perceiving from coordinates.
    """
    from rdkit import Chem

    m = _labelled(mol)
    inchi = Chem.MolToInchi(m, options='/FixedH')
    heavy = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() != 1]
    pos = {old: j for j, old in enumerate(heavy)}
    rank = list(Chem.CanonicalRankAtoms(Chem.RemoveAllHs(m), breakTies=False,
                                        includeChirality=False))
    atoms = sorted((rank[pos[a.GetIdx()]], a.GetProp('_CIPCode')) for a in m.GetAtoms()
                   if a.GetIdx() in pos and a.HasProp('_CIPCode'))
    bonds = sorted((tuple(sorted((rank[pos[b.GetBeginAtomIdx()]],
                                  rank[pos[b.GetEndAtomIdx()]]))), b.GetProp('_CIPCode'))
                   for b in m.GetBonds() if b.HasProp('_CIPCode')
                   and b.GetBeginAtomIdx() in pos and b.GetEndAtomIdx() in pos)
    return inchi, tuple(atoms), tuple(bonds)


def _labelled(mol):
    """A conformer-free copy of ``mol``, N tags off, CIP-labelled, unlabelled tags dropped."""
    from rdkit import Chem
    from rdkit.Chem import rdCIPLabeler

    m = _strip_n_tags(Chem.Mol(mol))
    m.RemoveAllConformers()
    for x in (*m.GetAtoms(), *m.GetBonds()):
        x.ClearProp('_CIPCode')
    rdCIPLabeler.AssignCIPLabels(m)
    for a in m.GetAtoms():
        if a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED and not a.HasProp('_CIPCode'):
            a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    for b in m.GetBonds():
        if b.GetStereo() != Chem.BondStereo.STEREONONE and not b.HasProp('_CIPCode'):
            b.SetStereo(Chem.BondStereo.STEREONONE)
    return m


def _n_stereo_marks(smiles: str) -> int:
    return smiles.replace('@@', '@').count('@') + smiles.count('/') + smiles.count(chr(92))


def smiles_identity(smiles: str) -> tuple:
    """``stereo_identity`` of a SMILES as parsed -- the isomer its tags name."""
    with legacy_stereo():
        return stereo_identity(_parse(smiles))


def stereoisomer_classes(smiles: str, max_isomers: int = 1024) -> list:
    """``[(name, identity, other_names)]``: every stereoisomer of ``smiles``, sorted by name.

    Every assignment of the open elements (``onlyUnassigned``: an element the input already
    specifies is kept), N tags stripped, written as a canonical tagged SMILES and re-written
    once more; then the strings are GROUPED BY ``stereo_identity`` OF THE STRING AS PARSED,
    not by the string. Each group is one stereoisomer: its ``name`` is the string with the
    fewest stereo marks (then the lexically smallest), ``other_names`` are the rest --
    strings that name the SAME isomer, recorded rather than built twice (QM9 40520; both
    spurious "cis/trans" strings of a 1,3-disubstituted bicyclo[1.1.1]pentane). Every
    identity is read from a parsed string, the form the member
    is built from and checked against, never from the assignment object: on some cages
    ``rdCIPLabeler`` labels the assignment and its own SMILES differently although a
    chirality-respecting isomorphism maps one onto the other (QM9 105841), and grouping on
    the assignment split one isomer into two groups.

    Lossy in one known direction, and no worse than the strings alone: where the writer
    turns an assignment into a SMILES of ANOTHER isomer (QM9 32293), that assignment's own
    isomer is represented only if some other assignment writes it faithfully.

    TERMINATION: the assignments are counted first (``GetStereoisomerCount``, 2**n) and a
    count above ``max_isomers`` raises rather than being truncated to a random subset, which
    is what RDKit does past its cap.
    """
    from rdkit import Chem
    from rdkit.Chem.EnumerateStereoisomers import (EnumerateStereoisomers,
                                                   GetStereoisomerCount,
                                                   StereoEnumerationOptions)

    opts = StereoEnumerationOptions(onlyUnassigned=True, unique=True, maxIsomers=0)
    with legacy_stereo():
        m = stereo_mol(smiles)
        n = int(GetStereoisomerCount(m, options=opts))
        if n > int(max_isomers):
            raise ValueError(f'{smiles}: {n} stereo assignments exceed max_isomers='
                             f'{max_isomers}; refusing a truncated enumeration')
        strings = set()
        for iso in EnumerateStereoisomers(m, options=opts):
            iso = _strip_n_tags(Chem.Mol(iso))
            strings.add(Chem.MolToSmiles(_parse(Chem.MolToSmiles(Chem.RemoveHs(iso)))))
        classes = {}
        for s in strings:
            idt = stereo_identity(_parse(s))
            classes.setdefault(idt, set()).add(s)
            # the same SMILES with the tags no CIP label backs dropped, when it still parses
            # to this isomer: a 1,3-disubstituted bicyclo[1.1.1]pentane is then named without
            # the bridgehead tags legacy writes on it, which the encoder would read as parity
            bare = Chem.MolToSmiles(_labelled(_parse(s)))
            if bare != s and stereo_identity(_parse(bare)) == idt:
                classes[idt].add(bare)
    out = []
    for k, v in classes.items():
        # fewest stereo marks, then lexical: a function of the set, not of the walk
        name = min(v, key=lambda x: (_n_stereo_marks(x), x))
        out.append((name, k, sorted(v - {name})))
    return sorted(out)


def enumerate_stereoisomers(smiles: str, max_isomers: int = 1024) -> list:
    """Every stereoisomer of ``smiles``, one tagged SMILES each, sorted.

    ``stereoisomer_classes``'s names: one string per stereoisomer by ``stereo_identity``, so
    the list is a function of the molecule rather than of RDKit's walk order, and a tagged
    input enumerates to one string.
    """
    return [name for name, _, _ in stereoisomer_classes(smiles, max_isomers)]


def unspecified_stereo(smiles: str) -> list:
    """``[]`` when ``smiles`` names ONE stereoisomer, else the isomers it leaves open.

    JUDGED BY THE ENUMERATOR ITSELF, so "fully specified" means exactly what enumeration
    produced. An earlier ``FindPotentialStereo(cleanIt=True)`` check disagreed with it on
    centres that cannot be stereocentres given the rest of the molecule: both bridgeheads of
    a bicyclo[1.1.1]pentane, the apex of a trans-fused bicyclo[n.1.0] ring. It refused
    molecules with no open element at all (30 of 5,000 random QM9 molecules whole, 101 in
    part), none for the tetrahedral-N reason it was once blamed on.
    """
    classes = stereoisomer_classes(smiles)
    return [] if len(classes) <= 1 else [c[0] for c in classes]


def _tagged_from_3d(mol3d):
    """A copy of ``mol3d`` with every tag replaced by what its geometry says, N tags off."""
    from rdkit import Chem

    m = Chem.Mol(mol3d)
    Chem.AssignStereochemistryFrom3D(m)
    return _strip_n_tags(m)


def realised_isomer(mol3d) -> str:
    """The stereoisomer an embedded molecule IS, re-perceived from 3D, as a SMILES to READ.

    ``AssignStereochemistryFrom3D``, then the canonical isomeric SMILES of the H-stripped
    copy (RemoveHs keeps an H that defines double-bond stereo, so an imine keeps its [H]).
    For messages and tables: the comparison against a requested isomer goes through
    ``stereo_identity``, because this string can name a different isomer on a cage.
    """
    from rdkit import Chem

    with legacy_stereo():
        m = _tagged_from_3d(mol3d)
        return Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(Chem.RemoveHs(m))))


def constitution_smiles(smiles: str) -> str:
    """The stereo-free canonical SMILES, explicit H dropped (the embedding probe's input)."""
    from rdkit import Chem

    m = _parse(smiles)
    Chem.RemoveStereochemistry(m)
    return Chem.MolToSmiles(Chem.RemoveHs(m))


@functools.lru_cache(maxsize=4096)
def untagged_realisations(constitution: str, seed: int, n_seeds: int = 4) -> frozenset:
    """``stereo_identity`` of every isomer ETKDG realises for the UNTAGGED constitution.

    One ETKDGv3 embedding per seed in ``seed .. seed + n_seeds - 1`` (the member's own seed
    first), MMFF-relaxed when MMFF types it, re-perceived from 3D. Bounded: ``n_seeds``
    embeddings, each bounded inside RDKit. Cached per constitution, so the many isomers of
    one cage cost one probe.

    WHY. ETKDG with chirality constraints fails on isomers a strained cage does realise:
    ``CC1C2C3CCC2C13`` (QM9 105841) embeds untagged at every seed 0-7, always as
    ``C[C@H]1[C@H]2[C@@H]3CC[C@H]2[C@H]13``, and that SMILES embeds at none of seeds 0-7,
    with or without random coordinates. This separates those refusals
    (``embed_failed_realisable``) from isomers nothing has shown to exist.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    out = set()
    for s in range(int(seed), int(seed) + int(n_seeds)):
        m = Chem.AddHs(_parse(constitution))
        p = AllChem.ETKDGv3()
        p.randomSeed = s
        if AllChem.EmbedMolecule(m, p) != 0:
            continue
        try:
            AllChem.MMFFOptimizeMolecule(m, maxIters=2000)
        except Exception:                                    # noqa: BLE001 - probe only
            pass
        with legacy_stereo():
            out.add(stereo_identity(Chem.RemoveHs(_tagged_from_3d(m))))
    return frozenset(out)


# ------------------------------------------------------------------ one member


@dataclass
class Member:
    """A molecule that passed every check: its chart, and its member-width condition."""
    identifier: str
    smiles: str
    energy: object
    condition: object
    atoms: Optional[np.ndarray] = None       # free_dof_atom_index, when embedded
    mask: Optional[np.ndarray] = None


def _bond_graph_mismatch(energy):
    """``(missing, extra)``: RDKit bonds absent from the chart's graph, and the converse.

    The chart -- tree, nonbonded pairs, closures -- is built on ``infer_bond_index`` over the
    ETKDG+MMFF reference, while ``ff_from_mmff`` takes its bonded terms from the RDKit graph.
    When the reference stretches a bond past the perception cutoff the two disagree: the
    pair then gets nonbonded terms ON TOP of its bond term, and the member is not MMFF94 of
    its molecule. Read off the member's own ``bond_index_slot`` (mapped back through
    ``spec.perm``), so this is the graph the member was built on, not a re-perception.
    """
    perm = np.asarray(energy.spec.perm)
    inferred = {tuple(sorted((int(perm[a]), int(perm[b]))))
                for a, b in np.asarray(energy.bond_index_slot).reshape(2, -1).T}
    rdkit = {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx())))
             for b in energy.mol.GetBonds()}
    return sorted(rdkit - inferred), sorted(inferred - rdkit)


def build_member(smiles: str, identifier: str, energy_kw: dict, *, bundle=None,
                 carrier: bool = False, check: bool = True) -> Member:
    """One molecule's chart and condition graph, or ``MemberRefused`` naming why not.

    THE ORDER IS CONSTITUTION FIRST, ISOMER LAST. The refusals that are properties of the
    bond graph under ``force_field: mmff`` (atom count, MMFF typing) come first, so every
    stereoisomer of a molecule fails them identically and the molecule's recorded reason is
    the constitution-level blocker rather than whichever isomer happened to be tried first
    (build_conformer_set.CONSTITUTION_CODES). The builder's stereo checks read the embedded
    reference, so they come after it exists. ANY exception, from any step, leaves as
    ``MemberRefused``.
    """
    try:
        return _build_member(smiles, identifier, energy_kw, bundle=bundle, carrier=carrier,
                             check=check)
    except MemberRefused:
        raise
    except Exception as exc:                                   # noqa: BLE001 - classified
        raise MemberRefused(classify_failure(exc), f'{type(exc).__name__}: {exc}') from None


def _embed_failure(smiles: str, seed: int, message: str):
    """``(code, message)`` for a construction that failed to embed: realisable or not."""
    try:
        want = smiles_identity(smiles)
        seen = untagged_realisations(constitution_smiles(smiles), int(seed), PROBE_SEEDS)
    except Exception as exc:                                   # noqa: BLE001 - probe only
        return 'embed_failed', f'{message} (realisability probe raised {exc!r})'
    seeds = f'seeds {seed}..{int(seed) + PROBE_SEEDS - 1}'
    if want in seen:
        return 'embed_failed_realisable', (
            f'{message}; the untagged constitution embeds AS this stereoisomer ({seeds}), '
            f'so the isomer exists and the chirality-constrained embedding is what failed')
    return 'embed_failed', (f'{message}; not among the {len(seen)} isomer(s) the untagged '
                            f'constitution realises ({seeds})')


def _build_member(smiles, identifier, energy_kw, *, bundle, carrier, check) -> Member:
    try:
        energy = ConformerTorsions(smiles=smiles, device='cpu', **energy_kw)
    except Exception as exc:                                   # noqa: BLE001 - classified
        code, msg = classify_failure(exc), f'{type(exc).__name__}: {exc}'
        # A CHART REFUSAL CARRIES ITS CAUSE AS A CODE (conformer_torsions.ChartRefused), which
        # beats matching its prose; the chart's 'incomplete_chart' is this table's
        # 'incomplete_chart_full'
        chart_code = _CHART_CODES.get(getattr(exc, 'code', None))
        if chart_code is not None:
            code = chart_code
        if code == 'embed_failed':
            code, msg = _embed_failure(smiles, energy_kw.get('seed', 0), msg)
        raise MemberRefused(code, msg) from None
    level = energy.level

    # THE CHART AND THE COLLECTIVE MAP MUST AGREE ON HOW MANY COLUMNS THERE ARE -- at
    # `torsion` only. `energy.mask` carries one column per ROTATABLE AXIS while `_M` and
    # `data_ndim` carry one per SURVIVING axis; they disagree whenever an axis is dropped as
    # degenerate (measured: 11 of 400 QM9 molecules, every one an alkyne), and
    # `_state_columns` would then emit a column the state cannot hold. Above `torsion` the
    # two counts differ by construction (butanol: mask 2 against _M 39 at `full`), so the
    # comparison means nothing there.
    if level == 'torsion' and int(energy.mask.shape[1]) != int(energy._M.shape[1]):
        raise MemberRefused('chart_defect',
                            f'mask has {int(energy.mask.shape[1])} columns, _M has '
                            f'{int(energy._M.shape[1])} (alkyne?)')
    # A CONSTRAINED CHART MUST NOT ENTER A FILE LABELLED `full`. The energy refuses one
    # unless energy_config.allow_constrained opts in; a file is refused regardless, since its
    # consumers read `level: full` as 3N-6.
    if level == 'full' and int(getattr(energy, 'constrained_rows', 0)):
        raise MemberRefused('incomplete_chart_full',
                            f'CONSTRAINED at full: d={energy.data_ndim} against '
                            f'3N-6={3 * energy.spec.n_atoms - 6}, {energy.constrained_rows} '
                            f'row(s) held at a linear centre')

    missing, extra = _bond_graph_mismatch(energy)
    if missing or extra:
        raise MemberRefused('graph_broken',
                            f"geometry-inferred bond graph differs from RDKit's: missing "
                            f'{missing}, extra {extra} (RDKit atom numbering)')

    # THE LOCK MUST BE EXACTLY 0 ON ITS OWN ISOMER'S THERMAL ENSEMBLE, checked per molecule
    # because the census that calibrated the indicator is a sample of QM9 isomers, and a
    # reference in a metastable basin is seen by nothing else (stereo_lock.py). An MMFF94
    # chain independent of the chart (stereo_lock.mmff_thermal_samples); a molecule on which
    # the lock fires is refused under the in-band code, since the cause is the same -- an
    # indicator the correct isomer's own motion carries into its band.
    if float(energy_kw.get('stereo_coeff', 0) or 0) > 0:
        from energies.stereo_lock import thermal_check
        tc = thermal_check(energy, seed=int(energy_kw.get('seed', 0) or 0))
        if tc['fired']:
            raise MemberRefused('stereo_lock_in_band',
                                f"thermal check: the lock fired on {tc['fired']} of {tc['n']} "
                                f"MMFF94 samples of its own isomer "
                                f"(min s*v - lo {tc['min_excess']:.3f})")

    layout1 = None
    if carrier:
        # ADMISSIBILITY ASKED OF THE LAYOUT ITSELF, one member at a time, rather than
        # re-deriving its rule here: whatever CarrierLayout refuses is refused for this
        # molecule alone. It places a transverse column in the theta region, so it refuses
        # only a block code with no carrier region -- a new chart code the layout was not
        # taught, recorded as 'other' with its message.
        from energies.conformer_carrier import CarrierLayout
        layout1 = CarrierLayout({identifier: energy})

    mol = condition_from_energy(energy, identifier=identifier)
    if check:
        try:
            check_state_convention(mol, energy)
            if layout1 is not None:
                # the carrier map, checked here in a one-member layout so a member-local
                # defect is THIS member's refusal; the set-level check after padding into
                # the shared layout can then fail only on a layout defect
                from energies.conformer_carrier import (carrier_pad_condition,
                                                        check_carrier_convention)
                check_carrier_convention(carrier_pad_condition(mol, layout1, identifier,
                                                               energy),
                                         layout1, identifier, energy)
        except AssertionError as exc:
            raise MemberRefused('convention_check', str(exc)) from None

    _check_stereo(smiles, energy)

    atoms = mask = None
    if bundle is not None:
        # TREE ORDER, via spec.perm, and asserted against spec.z inside `embed`. The encoder
        # and the conformer path order atoms differently (heavy-then-hydrogen against tree
        # placement), and attaching encoder-order rows to a conformer batch would condition
        # every atom on another atom with no shape error to catch it. The SMILES is the
        # stereo-TAGGED one, so the encoder's CIP parity channel is live.
        h, g, _ = encoder_cache.embed(bundle, smiles, perm=energy.spec.perm,
                                      z_tree=energy.spec.z)
        mol.embedding = g[None, :].to(torch.get_default_dtype())
        mol.atom_embedding = h.to(torch.get_default_dtype())
        atoms, mask = free_dof_atom_index(energy)
    return Member(identifier, smiles, energy, mol, atoms, mask)


def _check_stereo(smiles: str, energy):
    """Refuse a SMILES that leaves stereo open, or a reference that realised another isomer.

    An N tag is refused outright: 3D perception cannot see it, so the energy could not be
    shown to hold it, and a tag nothing enforces is a label that lies. The reference is
    compared by ``stereo_identity`` against the SMILES AS PARSED -- no canonical rewrite on
    either side, since the rewrite can itself change the isomer.
    """
    from rdkit import Chem

    m = _parse(smiles)
    if any(a.GetAtomicNum() == 7 and a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
           for a in m.GetAtoms()):
        raise MemberRefused('stereo_n_tagged',
                            f'{smiles} tags a tetrahedral N; N configuration is not '
                            f'perceivable from 3D, so it cannot be verified or pinned')
    open_ = unspecified_stereo(smiles)
    if open_:
        raise MemberRefused('stereo_unspecified',
                            f'{smiles} leaves stereo open: it enumerates to {len(open_)} '
                            f'stereoisomers ({", ".join(open_[:4])}{", ..." * (len(open_) > 4)});'
                            f' ETKDG would choose by seed and the identifier would not say '
                            f'which. Pass a stereo-tagged SMILES, or build the set with '
                            f'build_conformer_set.py, which enumerates them')
    want = smiles_identity(smiles)
    with legacy_stereo():
        # H-stripped, so the labeller reads the same atoms in the same order as the parse
        # it is compared with (the member's mol is that parse with hydrogens appended)
        got = stereo_identity(Chem.RemoveHs(_tagged_from_3d(energy.mol)))
    if got != want:
        part = 'InChI' if got[0] != want[0] else 'CIP labels'
        raise MemberRefused('stereo_verify_failed',
                            f'reference re-perceives as {realised_isomer(energy.mol)}, '
                            f'requested {smiles} ({part} differ)')


# ------------------------------------------------------------------ prior draws


def draw_prior_states(energy, n: int, internal_prior: Path, fatten: float, seed: int):
    """``[n, k]`` TORSION-TIER states from the fitted InternalPrior, or uniform on the torus.

    ``build_prior_states.draw_states`` emits one column per ROTATABLE AXIS, which is the
    state only at level `torsion`; anywhere else it is refused (``main`` routes those levels
    through ``energies/conformer_prior_draw.draw_member_prior``). Uniform on the torus is the
    correct dumb prior for a torsion (maximum entropy on the space), it is just a much
    weaker one, and it is announced.
    """
    if energy.level != 'torsion':
        raise ValueError(f'{energy.smiles}: draw_prior_states is the torsion-tier draw '
                         f'(one column per rotatable axis); level {energy.level!r} has '
                         f'{energy.data_ndim} columns. Use conformer_prior_draw.'
                         f'draw_member_prior')
    rng = np.random.default_rng(seed)
    if internal_prior is not None and Path(internal_prior).exists():
        from build_prior_states import draw_states, fit_or_load

        prior = fit_or_load(Path(internal_prior), [], fatten)
        states, n_uniform = draw_states(energy, prior, n, rng)
        if n_uniform:
            print(f"  {energy.smiles}: {n_uniform}/{energy.data_ndim} dimensions had no "
                  f"fitted torsion type and fell through to uniform")
        return states.to(energy.dtype)

    print(f"  {energy.smiles}: no fitted InternalPrior at {internal_prior}; drawing "
          f"UNIFORM on the torus ({n} states). This is the max-entropy dumb prior, not a "
          f"failure -- but it is much weaker than the fitted one")
    return torch.as_tensor(rng.uniform(-1.0, 1.0, (n, energy.data_ndim)), dtype=energy.dtype)


def report_rejections(skipped, n_in: int):
    """Print every refusal, grouped by code. ``skipped`` is ``[(smiles, ident, code, msg)]``."""
    if not skipped:
        return
    print('')
    print(f'{len(skipped)} of {n_in} molecules refused:')
    why = {}
    for smi, _, code, msg in skipped:
        why.setdefault(code, []).append((smi, msg))
    for code, rows in sorted(why.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        print(f'   {len(rows):4d}  {code}')
        for smi, msg in rows:
            print(f'           {smi}: {msg[:160]}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smiles", nargs="+", default=["CCCCO"])
    ap.add_argument("--identifiers", nargs="*", default=None,
                    help="one per SMILES; defaults to the SMILES themselves. train.py "
                         "resolves condition identity through this string alone")
    ap.add_argument("--out", type=Path, default=artifact("conformer_conditions.pt"))
    ap.add_argument("--prior-out", type=Path, default=None)
    ap.add_argument("--n-prior", type=int, default=4000,
                    help="states per molecule in the prior file")
    ap.add_argument("--internal-prior", type=Path, default=None,
                    help="a fitted InternalPrior .pt (see build_prior_states.py). Default: "
                         "energy_config.internal_prior_path of --config, if given")
    ap.add_argument("--fatten", type=float, default=0.15)
    ap.add_argument("--config", type=Path, default=None,
                    help="take the energy kwargs from this run config's energy_config, "
                         "filtered by the ConformerTorsions signature exactly as the run "
                         "filters them. Flags below that are given override it, and are "
                         "printed")
    # None = not given: the value then comes from --config, else ConformerTorsions' default
    ap.add_argument("--epsilon", type=float, default=None)
    ap.add_argument("--min-separation", type=int, default=None)
    ap.add_argument("--scale-14", type=float, default=None)
    ap.add_argument("--lj-k-factor", type=float, default=None)
    ap.add_argument("--delta-r-max", type=float, default=None)
    ap.add_argument("--delta-theta-max", type=float, default=None)
    ap.add_argument("--r-floor", type=float, default=None)
    ap.add_argument("--theta-floor", type=float, default=None)
    ap.add_argument("--energy-clip", type=float, default=None,
                    help="energy_config.energy_clip. Baked prior energies pass through it, "
                         "so a prior built without the run's clip stores rewards the run "
                         "would not compute")
    ap.add_argument("--mmff-reference", dest="mmff_reference", action="store_true",
                    default=None)
    ap.add_argument("--no-mmff-reference", dest="mmff_reference", action="store_false")
    ap.add_argument("--include-trivial-rotations", action="store_true", default=None)
    ap.add_argument("--force-field", default=None, choices=("reference", "mmff"),
                    help="default: --config's, else 'mmff'. It selects how linear centres "
                         "are flagged, which can change the chart -- a file built under a "
                         "different one can disagree with the run's energy")
    ap.add_argument("--seed", type=int, default=None,
                    help="the ETKDG embedding seed (energy_config.seed); also offsets the "
                         "per-molecule prior seeds")
    ap.add_argument("--stereo-coeff", type=float, default=None,
                    help="energy_config.stereo_coeff (default: --config's, else 0). Every "
                         "graph records it (ctree_stereo_coeff) and a run whose energy "
                         "differs refuses the file: it decides the baked energies (the lock "
                         "sits inside them) and which rows the per-coordinate features mark "
                         "as held. Above 0, every SMILES must name one stereoisomer, and each "
                         "molecule must also pass a thermal check of the lock "
                         "(stereo_lock.thermal_check)")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--no-check", action="store_true",
                    help="skip the graph-vs-energy geometry check (don't)")
    ap.add_argument("--level", default=None,
                    choices=("torsion", "dihedral", "flex", "full"),
                    help="which internal DoF the state drives; default --config's, else "
                         "'full'. `full` is every one of them; the narrower tiers freeze the "
                         "rest at the reference conformer, which is a DIFFERENT distribution, "
                         "not a coarser view of the same one")
    ap.add_argument("--k", type=int, default=None,
                    help="keep only molecules with this state dimension. Default: take k from "
                         "the first molecule that builds. A plain conditions file is ONE k")
    ap.add_argument("--carrier", action="store_true",
                    help="write a MIXED-k set in the width-K carrier layout "
                         "(energies/conformer_carrier.py) instead of filtering to one k. "
                         "The run builds the same layout from the same member set")
    ap.add_argument("--encoder-ckpt", type=Path, default=None,
                    help="bake a frozen molecular embedding onto every entry, so the policy "
                         "can be conditioned on molecular identity. Writes per-graph "
                         "`embedding` (pooled, 2*hidden) and per-atom `atom_embedding` "
                         "(hidden, in TREE order). Enable with model.embedding_conditioning "
                         "and set embedding_conditioning_dim to the printed width")
    ap.add_argument("--rejections-out", type=Path, default=None,
                    help="also write the refusals as a TSV (smiles, identifier, code, message)")
    args = ap.parse_args(argv)

    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(args.threads)

    # STRIPPED. A SMILES list piped in from a CRLF file carries a trailing carriage
    # return into the identifier, which then keys the buffers, the mol_id registry and
    # the per-molecule energy table -- all of which match by string equality. The
    # contamination is invisible until one lookup uses a stripped key and raises a
    # KeyError three layers away.
    args.smiles = [s.strip() for s in args.smiles if s.strip()]
    if args.identifiers:
        args.identifiers = [s.strip() for s in args.identifiers if s.strip()]
    identifiers = args.identifiers or args.smiles
    if len(identifiers) != len(args.smiles):
        raise SystemExit(f"{len(args.smiles)} SMILES against {len(identifiers)} identifiers")

    ff, ec = ({}, {}) if args.config is None else energy_kwargs_from_config(args.config)
    flags = dict(epsilon=args.epsilon, min_separation=args.min_separation,
                 scale_14=args.scale_14, lj_k_factor=args.lj_k_factor,
                 delta_r_max=args.delta_r_max, delta_theta_max=args.delta_theta_max,
                 r_floor=args.r_floor, theta_floor=args.theta_floor,
                 energy_clip=args.energy_clip, mmff_reference=args.mmff_reference,
                 include_trivial_rotations=args.include_trivial_rotations, seed=args.seed,
                 force_field=args.force_field, level=args.level,
                 stereo_coeff=args.stereo_coeff)
    given = {k: v for k, v in flags.items() if v is not None}
    if args.config is not None and given:
        print(f'overriding {args.config} energy_config with {given}')
    ff.update(given)
    # FULL / MMFF BY DEFAULT: the production target. `torsion` and 'reference' were the old
    # defaults, and a file built without the flags then carried a chart the run disagreed with
    ff.setdefault('level', 'full')
    ff.setdefault('force_field', 'mmff')
    print(f'energy kwargs: {ff}')
    internal_prior = args.internal_prior or ec.get('internal_prior_path')
    relax_steps = int(ec.get('prior_relax_steps', 0) or 0)

    bundle = None
    if args.encoder_ckpt is not None:
        bundle = encoder_cache.load_encoder(str(args.encoder_ckpt), device="cpu")
        print(f"encoder {bundle['arm']} @ {bundle['sha256'][:12]}  "
              f"hidden {bundle['hidden']}  ->  embedding_conditioning_dim: "
              f"{2 * bundle['hidden']}")

    # `kept` TRACKS THE SURVIVORS. Zipping the prior loop against the original
    # `identifiers` instead pairs survivor i with input i, so every row after the
    # first skip is labelled with the WRONG molecule -- no error, and the identifier
    # is what the buffers, the mol_id registry and the per-molecule energy all key on.
    members, skipped = [], []
    want_k = int(args.k) if args.k else None
    for smiles, ident in zip(args.smiles, identifiers):
        try:
            mb = build_member(smiles, ident, ff, bundle=bundle, carrier=args.carrier,
                              check=not args.no_check)
        except MemberRefused as exc:
            skipped.append((smiles, ident, exc.code, exc.message))
            continue
        k_here = int(mb.energy.data_ndim)
        if want_k is None:
            want_k = k_here
        if k_here != want_k and not args.carrier:
            skipped.append((smiles, ident, 'k_mismatch', f"k={k_here}, file is k={want_k}"))
            continue
        print(mb.energy.describe())
        members.append(mb)

    report_rejections(skipped, len(args.smiles))
    if args.rejections_out is not None:
        with open(args.rejections_out, 'w', encoding='utf-8', newline='\n') as f:
            f.write('smiles\tidentifier\tcode\tmessage\n')
            for smi, ident, code, msg in skipped:
                f.write(f'{smi}\t{ident}\t{code}\t{" ".join(msg.split())}\n')
    if not members:
        raise SystemExit("no molecules survived; nothing to write")

    conditions = [mb.condition for mb in members]
    kept = [mb.identifier for mb in members]
    dof_rows = [(mb.condition, mb.atoms, mb.mask) for mb in members if mb.atoms is not None]
    layout = None
    if args.carrier:
        # THE SAME LAYOUT THE RUN WILL BUILD: CarrierLayout is a pure function of the members'
        # block counts, and MultiConformerTorsions builds it from the same member set read off
        # this file. Every graph is re-expressed in it, and the reconstruction is checked
        # against each member's own chart after padding, not only before. Every member here
        # already passed a one-member CarrierLayout, so this cannot refuse the set.
        from energies.conformer_carrier import (CarrierLayout, carrier_pad_condition,
                                                check_carrier_convention)
        layout = CarrierLayout({mb.identifier: mb.energy for mb in members})
        print(layout.describe())
        R = max((mb.atoms.shape[1] for mb in members if mb.atoms is not None), default=1)
        padded = []
        for mb in members:
            pm = carrier_pad_condition(mb.condition, layout, mb.identifier, mb.energy,
                                       atoms=mb.atoms, mask=mb.mask, R=R)
            if not args.no_check:
                err = check_carrier_convention(pm, layout, mb.identifier, mb.energy)
                print(f"   {mb.identifier}: carrier graph and member chart agree to "
                      f"{err:.2e} A")
            padded.append(pm)
        conditions = padded
        dof_rows = []            # dof_atoms / dof_mask already written in carrier form

    if dof_rows:
        # R IS GLOBAL ACROSS THE FILE, not per molecule. A state column at level='torsion' is
        # collective and owns however many dihedral rows its bond drives -- 3 for one
        # molecule, 2 for the next -- and PyG concatenates these along dim 0, which requires
        # dim 1 to agree. Padding per molecule would fail at collation; padding to the file's
        # maximum makes the ragged dimension a property of the FILE, which is what the
        # consumer can reason about.
        R = max(a.shape[1] for _, a, _ in dof_rows)
        for mol, a, msk in dof_rows:
            pa = np.zeros((a.shape[0], R, a.shape[2]), dtype=np.int64)
            pm = np.zeros((a.shape[0], R), dtype=bool)
            pa[:, :a.shape[1]] = a
            pm[:, :msk.shape[1]] = msk
            # FLATTENED TO ONE ROW PER GRAPH, not left as [k, R, frame]. MolData's
            # `append_batch` classifies a tensor by matching dim 0 to the node or the graph
            # count; a [k, R, frame] field matches neither, so it is taken for SHARED
            # metadata and validated for equality across molecules -- which fails the moment
            # two molecules differ, i.e. always. As [1, k*R*frame] it is an ordinary
            # graph-level field that concatenates, replicates across prior states, and
            # survives the collate. `ConformerGFN.bind_molecular_conditioning` restores the
            # shape from k and MAX_FRAME.
            mol.dof_atoms = torch.as_tensor(pa).reshape(1, -1)
            mol.dof_mask = torch.as_tensor(pm).reshape(1, -1)
        print(f"   DoF atom frames: R = {R} (widest collective column in this file)")

    batch = collate_conditions(conditions)
    save_condition_file(batch, args.out)
    print(f"\nwrote conditions -> {args.out}  ({batch.num_graphs} graphs, "
          f"k = {int(batch.n_torsions[0])})")
    if bundle is not None:
        print("   embeddings baked. In the config set:")
        print("     embedding_conditioning: true")
        print(f"     embedding_conditioning_dim: {2 * bundle['hidden']}")

    if args.prior_out is None:
        return

    print(f"\nprior: {args.n_prior} states per molecule")
    level = ff['level']
    prior_obj = None
    if level != 'torsion':
        # LEVEL-AWARE, and the fitted prior is REQUIRED above `torsion`: the torsion-tier draw
        # emits one column per rotatable axis (it died at `full` on a 2-vs-39 shape), and a
        # uniform box over bond lengths and angles is not a prior
        if internal_prior is None or not Path(internal_prior).exists():
            raise SystemExit(f'--prior-out at level {level!r} needs the fitted InternalPrior '
                             f'(--internal-prior or --config energy_config.'
                             f'internal_prior_path); got {internal_prior}')
        prior_obj = torch.load(internal_prior, weights_only=False)
    from energies.conformer_prior_draw import draw_member_prior, member_prior_seed
    parts = []
    for mol, mb in zip(conditions, members):
        energy, ident = mb.energy, mb.identifier
        seed = member_prior_seed(ident, ff.get('seed', 0))
        if prior_obj is None:
            states = draw_prior_states(energy, args.n_prior, internal_prior,
                                       args.fatten, seed)
            # RAW energy, T = 1: prebuilt_sample_to_reward divides by the sampling
            # temperature itself (see conformer_data.bake_energies)
            e = bake_energies(energy, states)
        else:
            states, e, _ = draw_member_prior(energy, args.n_prior,
                                             np.random.default_rng(seed),
                                             relax_steps=relax_steps, prior=prior_obj)
        periodic = energy.periodic_dims
        if layout is not None:
            states = layout.to_carrier(ident, states)
            periodic = [b == 2 for b in layout.free_block]
        print(f"  {ident}: E median {e.median():+8.3f}  p10 {torch.quantile(e, 0.1):+8.3f}"
              f"  p90 {torch.quantile(e, 0.9):+8.3f}")
        # the mask is NOT optional: at `flex` and above the state carries linear r/theta
        # columns, and wrapping one folds a bond length to the opposite corner of the box
        parts.append(attach_states(mol, states, e, identifier=ident, periodic=periodic))

    prior = parts[0]
    for part in parts[1:]:
        prior = prior.append_batch(part)
    save_prior_file(prior, args.prior_out,
                    source="InternalPrior" if internal_prior else "uniform",
                    n_per_molecule=args.n_prior)
    print(f"\nwrote prior -> {args.prior_out}  ({prior.num_graphs} rows)")


if __name__ == "__main__":
    main()
