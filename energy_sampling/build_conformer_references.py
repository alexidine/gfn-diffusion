"""Offline per-molecule REFERENCE TABLE for conformer evaluation, keyed by identifier.

For every condition in a conditions file (``molecules_path`` form, ``save_condition_file``)
this computes, once, the target references the per-molecule physics block reads
(``conformer_eval_metrics.per_molecule_block``'s ``refs``):

  e_min        the energy FLOOR: the lowest raw potential a multi-start descent reaches
               in the member's own state box, with ``e_min_state`` (member width) beside
               it. An UPPER bound on the true minimum, like every multi-start search.
  basin_ref    ``prior_diagnostics.basin_reference`` -- the target's rotamer modes, their
               energies and which are accessible -- or its ``skipped`` form above
               ``max_modes`` modes, with the mode count kept either way.
  target_tc    ``conformer_eval_metrics.target_coupling`` of that basin table.

WHY OFFLINE. ``ConformerModeller._e_min`` runs ``tier_minimum`` lazily on
``self.energy_function`` with 256 starts drawn by ``torch.randperm`` over
``prior_dataset.x``. On a carrier set that energy is the dispatcher, whose chart methods
are refused, and the prior rows are a mixture of molecules; and because nothing is
checkpointed, the zero is redrawn on every requeue leg. The search costs seconds per
molecule, so a set of thousands cannot pay it at every launch. Computed here once, it is a
file the eval loads.

THE FLOOR SEARCH is ``prior_baselines.descend`` (Rprop, best point seen) from four kinds
of start, all clamped into the box so the floor is a state the sampler can reach:

  ref      the member's own reference conformer, state 0;
  etkdg    the molecule re-embedded at further ETKDG seeds from the CONDITION'S SMILES (its
           stereo tags enforced, its open elements left to the embedder), MMFF-optimised as
           the reference is, then MEASURED IN THE MEMBER'S OWN TREE (``measure`` ->
           ``state_from_dof``);
  uniform  uniform on the box (``prior_baselines.draw_uniform``);
  prior    fitted ``InternalPrior`` draws (``prior_baselines.draw_prior``).

THE DATABASE FLOOR (``--database DIR``) replaces the search: each condition's floor is its
lowest row in a build_conformer_database.py directory, measured into the member and re-scored
(``database_floor``); the stamp's ``floor`` and each entry's ``floor_source`` say which.

The ETKDG starts exist because the other three are one family: a search that draws more of
the same kinds cannot see a basin none of them reaches, so comparing 64 against 256 such
starts cannot measure adequacy. Re-embedded references do reach such basins. See
``DEFAULT_SEARCH`` for the measured depth.

THE STEREO PIN IS THE CONDITION'S. Each stereoisomer is a distinct condition, and the
condition's stereo-tagged SMILES says what is pinned (``condition_stereo``). ``member.mol``
is ``AddHs(MolFromSmiles(smiles))``, so a tetrahedral tag or a double-bond E/Z label in the
SMILES lands on a known atom or bond of the member; each such element is PINNED at the
configuration the reference conformer realises, and a reference that does not realise the
SMILES's tags is refused. At `full` every phi column wraps, so a descent from a uniform
start can end in ANOTHER stereoisomer -- a mirror image at the same energy, or a diastereomer
at a different one -- and nothing in the potential stops it until a stereo lock term exists.
So the floor is the lowest candidate whose configuration, perceived from its 3D geometry
(``AssignStereochemistryFrom3D``), agrees with the reference's at every pinned element.
Elements the SMILES leaves OPEN are free, as they are for the sampler:
  * a stereo-free SMILES (every QM9 conditions file before the stereo builder) pins nothing,
    and its floor is the lowest state of ANY stereoisomer (``stereo_pin`` 'none');
  * an implicit-H SMILES cannot write =NH imine E/Z, so that element is open unless the
    SMILES carries the defining [H] (``stereo_pin`` 'partial', counted in ``stereo_open``);
  * RDKit's 3D perception returns no tetrahedral N, so a SMILES that tags an N cannot be
    checked and is refused.
``e_min_unpinned`` keeps the lowest candidate of ANY stereo and ``n_below_other_stereo`` how
many candidates below the floor break the pin, so the two are never confused.

CURRENCY. ``e_min`` is RAW POTENTIAL at T = 1 -- ``bake_energies``' number, the currency of
the baked ``conformer_energy`` -- never the tracker's ``best_energy``, which is
``-log_r * T`` and carries the log-Jacobian. The ranking is made in that currency too, by
re-scoring every candidate through ``bake_energies``.

THE STAMP makes a stale table refusable. It records the conditions file's sha256 (the
identity a manifest records for the file) and its identifiers, the member-defining
ConformerTorsions arguments resolved against the signature defaults, the search and basin
parameters, the internal prior's sha256, and the code revisions as data.
``load_references`` refuses a table whose conditions sha256 or member-defining arguments
differ from the caller's, or that lacks an entry for any of the file's identifiers (unless
the caller accepts a partial table), and ``ReferenceTable.verify`` re-scores every
``e_min_state`` through the run's own member, refusing a gap above 1e-2 kcal/mol -- the
``_load_prior_dataset`` discipline, which catches a changed force field or reference
geometry that equal metadata would hide.

RESUME. Each finished molecule is written at once to ``<out>.parts/``; a rerun with the
same stamp skips those molecules, and a parts directory from a different stamp is refused.

BOUNDED. Per molecule the work is fixed: ``n_starts x steps`` descent steps, one stereo
perception per candidate at most, ``max_modes`` basin energies, and each ETKDG embedding
under ``ETKDG_TIMEOUT_S``. The parent's wait on the worker pool is bounded by
``--molecule-timeout`` between completions: when no molecule finishes in that window the
pool is terminated, finished molecules are kept, and the build exits non-zero naming the
unfinished ones.

    python build_conformer_references.py --conditions D:/sets/r1/conditions_train.pt \\
        --config configs/conformer_mk.yaml --workers 8

The table is written beside the conditions file (``<stem>.references.pt``) unless
``--out`` names another path. CPU only, float64.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import inspect
import json
import os
import shutil
import subprocess
import sys
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

#: /2: the floor is pinned at the elements the condition's SMILES specifies (it was pinned to
#: every element of the reference's perceived stereo), so a /1 table means something else
FORMAT = 'conformer_references/2'
_SEED_SALT = b'conformer_references/1'

#: ConformerTorsions arguments that cannot change what a member's potential, chart or basin
#: table MEANS, so a table built under a different value still describes the run. Every other
#: argument is member-defining, INCLUDING ones added later (a stereo lock's, say): an unknown
#: argument is compared, which is the safe direction.
NON_DEFINING_KWARGS = frozenset({
    'device', 'dtype',
    # the condition vector only; the potential takes T explicitly
    'temperature_conditioning', 'embedding_conditioning', 'embedding_conditioning_dim',
    'log_temperature_range',
    # the InternalPrior sampler only (sample_prior_states / ring_blocks): they move which
    # prior STARTS are drawn, not the potential the floor is a minimum of
    'ring_jitter_scale', 'ring_min_bank_rows', 'ring_mode_fill', 'ring_pop_temper',
})

#: The floor search's depth: starts per kind and descent steps per start (288 starts).
#: MEASURED 2026-09-26 with this module's own search (artifacts/conformer_references_
#: 2026-09-26/: calib_depth.py, analyze_depth.py, out/analyze_depth.txt; level full, mmff,
#: energy_clip 300) on 34 molecules where the WP09 review's re-embedded seeds had lowered a
#: floor and 64 random census members, each as its stereo-free QM9 condition (nothing pinned)
#: and as its reference isomer tagged (everything pinned; 9 of 98 do not re-embed tagged).
#: The yardstick is the lowest pin-keeping candidate of a ~2,100-start pool per condition --
#: these rows, 64 more seeds, 1,792 more draws and, untagged, 6 seeds per enumerated
#: stereoisomer -- so each miss is a lower bound (kcal/mol, hard set / random set):
#:                                 untagged                          tagged
#:   these 288 starts              max miss 0.00 / 0.11              max miss 0.30 / 0.01
#:   the same 256 draws, no seeds  > 1 on 12/34 and 0/64 (max 8.8)   > 1 on 5/29, 2/60 (max 13.0)
#:   2,048 draws, no seeds         > 1 on 2/34 and 0/64 (max 4.6)    > 1 on 0/29, 1/60 (max 2.0)
#: So depth is bought with re-embeddings, not box draws: eight times the draws does less than
#: the 32 seeds. At this depth the 66 enantiomer pairs among the tagged conditions agree to
#: 0.004 kcal/mol (mirror_pairs.py, out/mirror_pairs.txt), the owner's equal-log-Z check
#: restated on the floor.
DEFAULT_SEARCH = dict(n_seeds=32, n_uniform=127, n_prior=128, steps=150)

BASIN_MAX_MODES = 512
#: kcal/mol. The re-score bar of ConformerModeller._load_prior_dataset.
EMIN_RESCORE_TOL = 1e-2
#: Angstrom. A member rebuilt from the file's SMILES must reproduce the file's reference.
REF_POS_TOL = 1e-5
#: seconds per ETKDG embedding (RDKit EmbedParameters.timeout), and the failed embeddings
#: after which a molecule's remaining seeds are skipped. A strained stereoisomer embedded
#: with its tags enforced can fail at every seed (measured on QM9 cages whose seed-0
#: reference is such an isomer: 0 of 8 seeds embed with its stereo enforced), so the cap
#: bounds the cost of such a molecule at ETKDG_MAX_FAILURES x ETKDG_TIMEOUT_S instead of
#: n_seeds x it.
ETKDG_TIMEOUT_S = 10
ETKDG_MAX_FAILURES = 3


class StaleReferencesError(ValueError):
    """A reference table that does not describe the run it is being loaded into."""


class IncompleteReferencesError(StaleReferencesError):
    """A reference table without an entry for every identifier of its conditions file."""


# --------------------------------------------------------------------------- identity


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def molecule_seed(identifier: str) -> int:
    """Per-molecule RNG seed from the identifier alone, so a draw is the same on any worker,
    in any order, on any rerun."""
    d = hashlib.blake2b(_SEED_SALT + identifier.encode(), digest_size=8).digest()
    return int.from_bytes(d, 'little') % (2 ** 31 - 1)


def _plain(v):
    """JSON-stable form, so a YAML list and the signature's tuple compare equal."""
    if isinstance(v, (list, tuple)):
        return [_plain(a) for a in v]
    if isinstance(v, Mapping):
        return {str(k): _plain(a) for k, a in v.items()}
    if isinstance(v, np.generic):
        return v.item()
    return v


def _ct_parameters():
    from energies.conformer_torsions import ConformerTorsions
    return inspect.signature(ConformerTorsions.__init__).parameters


def member_kwargs(energy_config) -> dict:
    """The ConformerTorsions arguments a run builds each member with, from its energy_config.

    The same filter as ``ConformerModeller.init_energy_function``: keys that are not
    parameters of ConformerTorsions are crystal-route or trainer keys and are dropped;
    ``smiles`` is dropped because the members come from the conditions file.
    """
    cfg = dict(energy_config) if isinstance(energy_config, Mapping) else dict(vars(energy_config))
    params = _ct_parameters()
    return {k: v for k, v in cfg.items()
            if k in params and k not in ('self', 'smiles', 'device', 'dtype')}


def defining_energy(kwargs: Mapping) -> dict:
    """The member-defining arguments, RESOLVED against the signature defaults.

    An absent key means the code default, so a config that omits `seed` and one that says
    `seed: 0` describe the same members and must compare equal; comparing the raw dicts
    would refuse one against the other.
    """
    out = {}
    for name, p in _ct_parameters().items():
        if name in ('self', 'smiles') or name in NON_DEFINING_KWARGS:
            continue
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if name in kwargs:
            out[name] = _plain(kwargs[name])
        elif p.default is not inspect.Parameter.empty:
            out[name] = _plain(p.default)
        else:
            raise ValueError(f'energy kwargs omit {name!r}, which has no default')
    return out


def _git(args, cwd) -> str:
    try:
        r = subprocess.run(['git', *args], cwd=cwd, capture_output=True, text=True, timeout=30)
        return r.stdout.strip() if r.returncode == 0 else ''
    except Exception:                                    # noqa: BLE001 - recorded as unknown
        return ''


def code_revisions() -> dict:
    """Revisions AS DATA: which code built the table. Not compared at load (a table built at
    another commit may be exactly right); the re-score in ``verify`` is what proves it."""
    import rdkit

    here = Path(__file__).resolve().parent
    out = {'gfn': _git(['rev-parse', 'HEAD'], here) or 'unknown',
           'gfn_dirty': bool(_git(['status', '--porcelain', '--untracked-files=no'], here)),
           'rdkit': rdkit.__version__, 'torch': torch.__version__, 'numpy': np.__version__}
    try:
        import mxtaltools
        mxt = Path(mxtaltools.__file__).resolve().parent
        out['mxtaltools'] = _git(['rev-parse', 'HEAD'], mxt) or 'unknown'
        out['mxtaltools_dirty'] = bool(_git(['status', '--porcelain', '--untracked-files=no'],
                                            mxt))
    except Exception:                                    # noqa: BLE001 - recorded as unknown
        out['mxtaltools'] = 'unknown'
    return out


# ------------------------------------------------------------------------ conditions


def read_conditions(path) -> List[dict]:
    """``[{identifier, smiles, z, pos}]`` per distinct identifier, in file order.

    Read the way ``ConformerModeller._condition_set_molecules`` reads the set (the
    ``prior`` batch's ``identifier`` and ``smiles``), plus each graph's placement-order
    ``z`` and reference ``pos``, which the member rebuilt from the SMILES must reproduce.
    An identifier that names two different SMILES is refused: the run would keep the first
    and score the second's rows against it.
    """
    blob = torch.load(path, weights_only=False, map_location='cpu')
    batch = blob['prior'] if isinstance(blob, dict) and 'prior' in blob else blob
    idents, smiles = getattr(batch, 'identifier', None), getattr(batch, 'smiles', None)
    if idents is None or smiles is None:
        raise ValueError(f'{path}: the conditions batch carries no identifier/smiles lists')
    if isinstance(idents, str):
        idents, smiles = [idents], [smiles]
    z = batch.z.detach().cpu().numpy()
    pos = batch.pos.detach().cpu().double().numpy()
    ptr = getattr(batch, 'ptr', None)
    ptr = (np.asarray([0, len(z)]) if ptr is None else ptr.detach().cpu().numpy())
    if len(ptr) != len(idents) + 1:
        raise ValueError(f'{path}: {len(idents)} identifiers against {len(ptr) - 1} graphs')
    out: Dict[str, dict] = {}
    for g, (ident, smi) in enumerate(zip(idents, smiles)):
        if ident in out:
            if out[ident]['smiles'] != smi:
                raise ValueError(f'{path}: identifier {ident!r} names two molecules '
                                 f'({out[ident]["smiles"]!r} and {smi!r})')
            continue
        a, b = int(ptr[g]), int(ptr[g + 1])
        out[ident] = dict(identifier=str(ident), smiles=str(smi), z=z[a:b].copy(),
                          pos=pos[a:b].copy())
    return list(out.values())


# --------------------------------------------------------------------------- members


@contextmanager
def _float64():
    """float64 for the duration, and the caller's default restored after. A module-scope or
    un-restored default leaks into every later test in the same process."""
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(old)


def build_member(smiles: str, kwargs: Mapping, pos=None, z=None):
    """The member exactly as ``MultiConformerTorsions`` builds it (``ConformerTorsions(smiles=
    ..., **kw)``), on the CPU in float64.

    With ``pos`` (and ``z``), the conditions file's stored reference in placement order, the
    member is built FROM IT, as the run builds it (``ConformerModeller.init_energy_function``):
    no embedding, so the member is the file's under any RDKit. A stored geometry that is not
    this molecule in the derived placement order raises ``ValueError``.
    """
    from energies.conformer_torsions import ConformerTorsions, rdkit_order_reference
    kwargs = dict(kwargs)
    rd = None
    if pos is not None:
        rd, _ = rdkit_order_reference(smiles, np.asarray(pos, dtype=np.float64),
                                      level=kwargs.get('level'), z=z)
    return ConformerTorsions(smiles=smiles, device='cpu', dtype=torch.float64,
                             reference_positions=rd, **kwargs)


def check_member_matches(member, z, pos, tol: float = REF_POS_TOL) -> float:
    """Refuse a member whose rebuilt reference is not the conditions file's.

    The run rebuilds each member from its SMILES, so an RDKit or seed difference gives a
    different reference conformer -- a different chart, in which a stored state means
    another geometry. Returns the max |dpos| in Angstrom.
    """
    mz = np.asarray(member.spec.z)
    if mz.shape != np.shape(z) or not np.array_equal(mz, np.asarray(z)):
        raise ValueError('member atoms (placement order) differ from the conditions file')
    gap = float(np.abs(member.ref_pos.detach().cpu().double().numpy() - np.asarray(pos)).max())
    if not gap <= tol:
        raise ValueError(f'member reference conformer differs from the conditions file by '
                         f'{gap:.3g} A (tol {tol:g}): the chart the file was built in is not '
                         f'the chart this build produces')
    return gap


# --------------------------------------------------------------------------- stereo


def rdkit_positions(member, states) -> np.ndarray:
    """``[B, n_atoms, 3]`` positions of member-width states, in RDKit atom order.

    ``build_positions`` returns PLACEMENT order; ``member.mol`` (and every RDKit call on it)
    is in RDKit order, and placement slot i holds RDKit atom ``spec.perm[i]``.
    """
    x = torch.as_tensor(states, dtype=member.dtype, device=member.device).reshape(-1, member.ndim)
    with torch.no_grad():
        # build_positions folds the batch into the atom dimension: [B * N, 3]
        pos = member.build_positions(x).detach().cpu().double().numpy().reshape(
            len(x), int(member.spec.n_atoms), 3)
    rd = np.empty_like(pos)
    rd[:, np.asarray(member.spec.perm)] = pos
    return rd


def _perceive(mol, rd_pos):
    """A copy of ``mol`` at ``rd_pos`` with its stereo perceived from that geometry."""
    from rdkit import Chem

    m = Chem.Mol(mol)
    m.GetConformer().SetPositions(np.ascontiguousarray(rd_pos, dtype=np.float64))
    Chem.AssignStereochemistryFrom3D(m)
    return m


def _isomer_smiles(m) -> str:
    """Canonical isomeric SMILES without the explicit H -- the identifier form of a tagged
    condition. RemoveHs keeps an H that defines double-bond stereo, so =NH E/Z survives."""
    from rdkit import Chem
    return Chem.MolToSmiles(Chem.RemoveHs(m))


def _stereo_elements(m):
    """``(atoms, bonds)``: indices of atoms with a tetrahedral tag and of double bonds with an
    E/Z label on ``m``."""
    from rdkit import Chem

    tetra = (Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
    atoms = tuple(a.GetIdx() for a in m.GetAtoms() if a.GetChiralTag() in tetra)
    bonds = tuple(b.GetIdx() for b in m.GetBonds()
                  if b.GetBondType() == Chem.BondType.DOUBLE
                  and b.GetStereo() not in (Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOANY))
    return atoms, bonds


def _labels(m, atoms, bonds) -> tuple:
    """The configuration of ``m`` at the given elements: chiral tags (relative to the atom's
    bond order, which every copy of one mol shares) then legacy E/Z labels."""
    return (tuple(int(m.GetAtomWithIdx(i).GetChiralTag()) for i in atoms)
            + tuple(int(m.GetBondWithIdx(j).GetStereo()) for j in bonds))


def _is_imine_nh(bond) -> bool:
    """An =NH double bond: one end is an N whose only other neighbour is an H."""
    for a, b in ((bond.GetBeginAtom(), bond.GetEndAtom()), (bond.GetEndAtom(), bond.GetBeginAtom())):
        others = [n for n in a.GetNeighbors() if n.GetIdx() != b.GetIdx()]
        if a.GetAtomicNum() == 7 and len(others) == 1 and others[0].GetAtomicNum() == 1:
            return True
    return False


def condition_stereo(member) -> dict:
    """The stereo pin a condition's SMILES defines on its member. See the module docstring.

    Returns ``atoms`` / ``bonds`` (the SPECIFIED elements, RDKit indices of ``member.mol``),
    ``target`` (their configuration on the reference conformer), ``pin`` ('none', 'partial'
    or 'full'), ``open_atoms`` / ``open_bonds`` / ``open_imine_nh`` (the elements RDKit
    perceives on the reference that the SMILES leaves unspecified), ``stereo`` (the
    reference's isomeric SMILES) and ``template`` (the explicit-H parse ETKDG re-embeds).

    The specified set is read off a FRESH parse of ``member.smiles``, not off ``member.mol``,
    whose tags other code re-assigns in place (``dof_features.atom_parity``). Refuses:
      * a parse whose atoms or bonds do not index ``member.mol`` (the member was not built
        as ``AddHs(MolFromSmiles(smiles))``);
      * RDKit in non-legacy stereo perception, where the SMILES's E/Z labels and the
        perceived ones need not be comparable -- the installed RDKit defaults to legacy and
        no module here changes it, so this fires only if that default moves;
      * a reference conformer that does not realise the SMILES's configuration at a
        specified element, including an element 3D perception returns nothing for (N).
    """
    from rdkit import Chem

    if not Chem.GetUseLegacyStereoPerception():
        raise RuntimeError('RDKit is set to non-legacy stereo perception; the pin compares the '
                           'SMILES\'s E/Z labels with legacy 3D-perceived ones. Restore it with '
                           'Chem.SetUseLegacyStereoPerception(True)')
    parsed = Chem.MolFromSmiles(member.smiles)
    if parsed is None:
        raise ValueError(f'{member.smiles!r} does not parse')
    tmpl = Chem.AddHs(parsed)
    mol = member.mol
    same = (tmpl.GetNumAtoms() == mol.GetNumAtoms() and tmpl.GetNumBonds() == mol.GetNumBonds()
            and all(a.GetAtomicNum() == mol.GetAtomWithIdx(a.GetIdx()).GetAtomicNum()
                    for a in tmpl.GetAtoms())
            and all((b.GetBeginAtomIdx(), b.GetEndAtomIdx())
                    == (mol.GetBondWithIdx(b.GetIdx()).GetBeginAtomIdx(),
                        mol.GetBondWithIdx(b.GetIdx()).GetEndAtomIdx()) for b in tmpl.GetBonds()))
    if not same:
        raise ValueError(f'{member.smiles!r}: AddHs(MolFromSmiles(smiles)) does not index the '
                         f'member\'s mol, so the SMILES\'s stereo tags cannot be placed on it')
    atoms, bonds = _stereo_elements(tmpl)
    k = int(member.ndim)
    ref = _perceive(mol, rdkit_positions(member, torch.zeros(1, k))[0])
    target = _labels(ref, atoms, bonds)
    want = _labels(tmpl, atoms, bonds)
    if target != want:
        bad = [f'atom {i}' for i, a, b in zip(atoms, target, want) if a != b]
        bad += [f'bond {mol.GetBondWithIdx(j).GetBeginAtomIdx()}-'
                f'{mol.GetBondWithIdx(j).GetEndAtomIdx()}'
                for j, a, b in zip(bonds, target[len(atoms):], want[len(atoms):]) if a != b]
        raise ValueError(f'{member.smiles!r}: the member\'s reference conformer is not the '
                         f'condition\'s stereoisomer at {", ".join(bad)} (3D perception gives '
                         f'{_isomer_smiles(ref)!r}; an element it returns no configuration for, '
                         f'such as a tetrahedral N, cannot be pinned)')
    p_atoms, p_bonds = _stereo_elements(ref)
    open_atoms = [i for i in p_atoms if i not in atoms]
    open_bonds = [j for j in p_bonds if j not in bonds]
    stereo = _isomer_smiles(ref)
    pin = 'none' if not (atoms or bonds) else ('partial' if open_atoms or open_bonds else 'full')
    if pin == 'full' and stereo != Chem.MolToSmiles(parsed):
        # every perceived element is specified and agrees, so the isomers must be one string
        raise ValueError(f'{member.smiles!r}: the reference perceives as {stereo!r}, which is '
                         f'not the condition\'s canonical {Chem.MolToSmiles(parsed)!r}')

    def pair(j):
        b = mol.GetBondWithIdx(j)
        return [b.GetBeginAtomIdx(), b.GetEndAtomIdx()]

    return dict(atoms=atoms, bonds=bonds, target=target, pin=pin, stereo=stereo,
                pinned={'atoms': list(atoms), 'bonds': [pair(j) for j in bonds]},
                open={'atoms': open_atoms, 'bonds': [pair(j) for j in open_bonds],
                      'imine_nh': sum(_is_imine_nh(mol.GetBondWithIdx(j)) for j in open_bonds)},
                template=tmpl)


def stereo_signatures(member, states) -> List[str]:
    """The isomeric SMILES perceived from each state's 3D geometry (``_isomer_smiles``).

    Two states of one member are the same stereoisomer exactly when these strings agree:
    the graph is fixed, so only the perceived tetrahedral and double-bond tags can differ.
    """
    return [_isomer_smiles(_perceive(member.mol, p)) for p in rdkit_positions(member, states)]


def stereo_labels(member, states, pin: Mapping) -> List[tuple]:
    """Each state's configuration at the PINNED elements; equal to ``pin['target']`` exactly
    when the state is the condition's stereoisomer there."""
    return [_labels(_perceive(member.mol, p), pin['atoms'], pin['bonds'])
            for p in rdkit_positions(member, states)]


def _pinned(pin: Mapping) -> bool:
    return bool(pin['atoms'] or pin['bonds'])


# --------------------------------------------------------------------------- search


def _embed(tmpl, seed: int, mmff: bool) -> Optional[np.ndarray]:
    """RDKit-order positions of one ETKDGv3 embedding of ``tmpl`` at ``seed``, MMFF-optimised
    when ``mmff`` (as ``ConformerTorsions`` optimises its reference); None when it fails.
    ETKDG enforces ``tmpl``'s stereo tags and picks each open element at random."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    m = Chem.Mol(tmpl)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.timeout = ETKDG_TIMEOUT_S
    if AllChem.EmbedMolecule(m, params) != 0:
        return None
    if mmff:
        AllChem.MMFFOptimizeMolecule(m, maxIters=2000)
    return m.GetConformer().GetPositions()


def _state_of_positions(member, tree, rd) -> torch.Tensor:
    """``[1, k]``: RDKit-order positions measured in the member's OWN tree (not a tree rebuilt
    from the new geometry, which may infer a different spanning tree). Not clamped.

    A dummy-frame row is measured against its dummy atom, as the member measures its ``ph0``
    (``member._dummy_t``, None when it has none): measured against the collinear real atom it
    reads noise, 0.19 in state units at the reference of CC#CC1CC1."""
    from mxtaltools.conformers.builder import measure

    pos = torch.as_tensor(np.asarray(rd)[np.asarray(member.spec.perm)], dtype=member.dtype,
                          device=member.device)
    r, th, ph = measure(tree, pos, dummy_frame=member._dummy_t)
    return member.state_from_dof(r.reshape(1, -1), th.reshape(1, -1), ph.reshape(1, -1))


def etkdg_starts(member, n_seeds: int, ref_seed: int, pin: Mapping, mmff: bool = True):
    """``([m, k] states, info)`` from re-embeddings at ETKDG seeds ref_seed+1 .. +n_seeds.

    The embedder is handed ``pin['template']``, the condition's own explicit-H parse, so a
    seed re-places the condition's stereo at every element the SMILES specifies and picks
    the open ones at random -- as a sampler without a lock may. An embedding that still
    comes back with another configuration at a pinned element is dropped and counted;
    after ``ETKDG_MAX_FAILURES`` failed embeddings the remaining seeds are skipped. Each
    geometry is measured in the member's own tree and clamped into the box.
    ``info['etkdg_seeds']`` lists the seed of every start returned, in order.
    """
    k = int(member.ndim)
    info = dict(n_etkdg=0, etkdg_failed=0, etkdg_other_stereo=0, etkdg_outside_box=0,
                etkdg_seeds=[])
    if n_seeds <= 0:
        return torch.zeros(0, k, dtype=member.dtype), info
    if member.collective:
        raise ValueError(f'ETKDG starts need state_from_dof, which level {member.level!r} '
                         f'has no row-wise inverse for; build with n_seeds=0')
    from mxtaltools.conformers.builder import collate

    tree = collate([member.spec], device=member.device)
    lin = member._lin_free_idx
    xs = []
    for s in range(int(ref_seed) + 1, int(ref_seed) + 1 + int(n_seeds)):
        rd = _embed(pin['template'], s, mmff)
        if rd is None:
            info['etkdg_failed'] += 1
            if info['etkdg_failed'] >= ETKDG_MAX_FAILURES:
                break
            continue
        if _pinned(pin) and (_labels(_perceive(member.mol, rd), pin['atoms'], pin['bonds'])
                             != pin['target']):
            info['etkdg_other_stereo'] += 1
            continue
        x = _state_of_positions(member, tree, rd)
        if lin.numel() and bool((x[:, lin].abs() > 1.0).any()):
            info['etkdg_outside_box'] += 1
        xs.append(x.clamp(-1.0, 1.0))
        info['etkdg_seeds'].append(int(s))
    info['n_etkdg'] = len(xs)
    return (torch.cat(xs) if xs else torch.zeros(0, k, dtype=member.dtype)), info


def search_starts(member, identifier: str, *, n_uniform: int, n_prior: int, n_seeds: int,
                  prior=None, ref_seed: int = 0, pin: Optional[Mapping] = None,
                  mmff: bool = True):
    """``(starts [n, k], kinds, info)``: ref, then etkdg, uniform and prior starts, clamped.

    A prior draw that raises (a member the fitted prior cannot type) is replaced by as many
    extra uniform starts, so the search keeps its depth, and the reason is recorded in
    ``info['prior_error']`` rather than dropped (``ReferenceTable.prior_fallbacks`` lists
    such molecules, since their starts are not the ones the depth was measured on).
    """
    from energies.prior_baselines import draw_prior, draw_uniform

    k = int(member.ndim)
    s = molecule_seed(identifier)
    if pin is None:
        pin = condition_stereo(member)
    parts, kinds = [torch.zeros(1, k, dtype=member.dtype)], ['ref']
    xe, info = etkdg_starts(member, n_seeds, ref_seed, pin, mmff=mmff)
    parts.append(xe.to(member.dtype))
    kinds += ['etkdg'] * len(xe)
    xp = torch.zeros(0, k, dtype=member.dtype)
    if n_prior > 0:
        if prior is None:
            raise ValueError('n_prior > 0 needs the fitted InternalPrior')
        try:
            xp = draw_prior(member, prior, int(n_prior), s + 1)[0].to(member.dtype)
        except Exception as exc:                         # noqa: BLE001 - recorded
            info['prior_error'] = f'{type(exc).__name__}: {str(exc)[:160]}'
    n_u = int(n_uniform) + int(n_prior) - len(xp)
    xu = draw_uniform(member, n_u, s).to(member.dtype)
    parts += [xu, xp]
    kinds += ['uniform'] * len(xu) + ['prior'] * len(xp)
    info.update(n_ref=1, n_uniform=len(xu), n_prior=len(xp))
    starts = torch.cat(parts).clamp(-1.0, 1.0)
    return starts, kinds, info


def floor_search(member, starts, kinds: Sequence[str], steps: int, pin: Mapping) -> dict:
    """The condition's floor over ``descend``'s candidates. See the module docstring on stereo.

    Every candidate is re-scored through ``bake_energies`` (raw potential, T = 1) and ranked
    in that currency; candidates are then perceived in increasing energy until one agrees
    with ``pin['target']`` at every pinned element, so at most one perception per candidate
    (none when nothing is pinned). ``e_min_stereo`` is the floor state's own isomeric SMILES.
    """
    from energies.conformer_data import bake_energies
    from energies.prior_baselines import descend

    bx, _ = descend(member, starts, int(steps))
    with torch.no_grad():
        bu = bake_energies(member, bx).detach().cpu().double().numpy()
    finite = np.isfinite(bu)
    if not finite.any():
        raise RuntimeError('no start reached a finite energy')
    order = [int(i) for i in np.argsort(np.where(finite, bu, np.inf), kind='stable')
             if finite[i]]
    chosen = None
    for rank, i in enumerate(order):
        if not _pinned(pin) or stereo_labels(member, bx[i:i + 1], pin)[0] == pin['target']:
            chosen = (rank, i)
            break
    if chosen is None:
        raise RuntimeError(f'none of {len(order)} finite candidates is the condition\'s '
                           f'stereoisomer at its pinned elements ({pin["pinned"]})')
    rank, i = chosen
    x = bx[i].detach().cpu().double().clone()
    if not bool((x.abs() <= 1.0 + 1e-12).all()):
        raise RuntimeError('the floor state lies outside the box')
    return dict(e_min=float(bu[i]), e_min_state=x, e_min_start_kind=kinds[i],
                e_min_stereo=stereo_signatures(member, x.reshape(1, -1))[0],
                e_min_unpinned=float(bu[order[0]]), n_below_other_stereo=int(rank),
                worst_start=float(bu[finite].max()), n_starts=int(len(bu)),
                n_nonfinite=int((~finite).sum()))


def basin_block(member, max_modes: int = BASIN_MAX_MODES) -> dict:
    """``basin_reference``, ``target_coupling`` and the mode counts, the skipped case included.

    ``n_modes`` is kept even when the enumeration is skipped (it is what a coverage-
    availability census stratifies on); ``n_accessible`` is None there because no mode was
    scored.
    """
    from energies.conformer_eval_metrics import target_coupling
    from energies.prior_diagnostics import basin_reference, rotamer_modes

    br = basin_reference(member, max_modes=int(max_modes))
    if 'skipped' in br:
        n_modes = int(np.prod([len(c) for _, c in rotamer_modes(member)]))
        n_acc = None
    else:
        n_modes, n_acc = len(br['combos']), int(np.asarray(br['accessible']).sum())
    return dict(basin_ref=br, target_tc=float(target_coupling(br)), n_modes=n_modes,
                n_accessible=n_acc, basin_skipped='skipped' in br)


def database_floor(member, identifier: str, rec, pin: Mapping, tol: float) -> dict:
    """The condition's floor from the conformer DATABASE: its lowest stored row, measured into
    ``member``'s own chart (``build_conformer_database.match_rows``: matched by identifier and
    member signature, re-scored against the stored energy within ``tol`` kcal/mol, refused
    otherwise). ``e_min`` is this member's re-score of that state, raw potential at T = 1.

    The row passed the database's own screen (every start whose configuration broke the pin
    was excluded before clustering), and is checked against this member's pin again here. The
    fields a search fills and the database does not record -- the lowest candidate of any
    stereo, how many candidates below the floor broke the pin, the start counts -- are None.
    """
    from build_conformer_database import match_rows
    from energies.conformer_data import bake_energies, wrap_state

    x, stored, info = match_rows(member, identifier, rec, 1, tol)
    if x is None:
        raise RuntimeError(f"database floor refused, {info['code']}: {info['message']}")
    x = wrap_state(x, member.periodic_dims)
    if not bool((x.abs() <= 1.0 + 1e-12).all()):
        raise RuntimeError('the database floor state lies outside the box')
    if _pinned(pin) and stereo_labels(member, x, pin)[0] != pin['target']:
        raise RuntimeError("the database floor state is not the condition's stereoisomer at "
                           f"its pinned elements ({pin['pinned']})")
    with torch.no_grad():
        e = float(bake_energies(member, x)[0])
    return dict(e_min=e, e_min_state=x[0].detach().cpu().double().clone(),
                e_min_start_kind='database',
                e_min_stereo=stereo_signatures(member, x)[0], e_min_unpinned=None,
                n_below_other_stereo=None, worst_start=None, n_starts=None, n_nonfinite=None,
                e_min_database=float(stored[0]), floor_rescore_gap=abs(e - float(stored[0])),
                ref_pos_gap_database=info['ref_pos_gap'])


def compute_entry(identifier: str, smiles: str, z, pos, kwargs: Mapping, search: Mapping,
                  prior=None, max_modes: int = BASIN_MAX_MODES, floor: str = 'search',
                  db_rec=None, db_tol: Optional[float] = None) -> dict:
    """One molecule's entry. float64 throughout; the caller's default dtype is restored.

    ``floor`` 'search' runs the multi-start floor search; 'database' takes the floor from
    ``db_rec``, the condition's database record (None when the database holds none, which
    raises) (``database_floor``). ``basin_ref`` is computed the same way under both.
    """
    from energies.conformer_data import bake_energies
    from energies.ring_metrics import ring_cycles

    if floor not in ('search', 'database'):
        raise ValueError(f'floor {floor!r}: search or database')
    cost = {}
    with _float64():
        t = time.perf_counter()
        member = build_member(smiles, kwargs, pos=pos, z=z)
        pos_gap = check_member_matches(member, z, pos)
        k = int(member.ndim)
        pin = condition_stereo(member)
        with torch.no_grad():
            u_ref = float(bake_energies(member, torch.zeros(1, k))[0])
        cost['build'] = time.perf_counter() - t

        if floor == 'database':
            info = {}
            cost['starts'] = 0.0
            t = time.perf_counter()
            floor_d = database_floor(member, identifier, db_rec, pin, float(db_tol))
            cost['descend'] = time.perf_counter() - t
        else:
            t = time.perf_counter()
            starts, kinds, info = search_starts(
                member, identifier, n_uniform=search['n_uniform'], n_prior=search['n_prior'],
                n_seeds=search['n_seeds'], prior=prior, ref_seed=int(kwargs.get('seed', 0)),
                pin=pin, mmff=bool(kwargs.get('mmff_reference', True)))
            cost['starts'] = time.perf_counter() - t

            t = time.perf_counter()
            floor_d = floor_search(member, starts, kinds, search['steps'], pin)
            cost['descend'] = time.perf_counter() - t

        t = time.perf_counter()
        basin = basin_block(member, max_modes)
        cost['basin'] = time.perf_counter() - t
        try:
            n_rings = int(len(ring_cycles(member)))
        except Exception:                                # noqa: BLE001 - recorded
            n_rings = -1
    cost['total'] = sum(cost.values())
    return dict(identifier=identifier, smiles=smiles, k=k, n_atoms=int(member.spec.n_atoms),
                n_rings=n_rings, stereo=pin['stereo'], stereo_pin=pin['pin'],
                stereo_pinned=pin['pinned'], stereo_open=pin['open'], u_ref=u_ref,
                ref_pos_gap=pos_gap,
                steps=int(search['steps']) if floor == 'search' else None,
                seed=molecule_seed(identifier), floor_source=floor,
                starts=info, **floor_d, **basin, cost_s=cost)


# ----------------------------------------------------------------------------- build

_WORKER: dict = {}


def _quiet(threads: int = 1):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    warnings.filterwarnings('ignore')
    torch.set_num_threads(int(threads))
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')


def _load_prior(path):
    if path is None:
        return None
    from energies.prior_baselines import load_prior
    return load_prior(str(path))[0]           # refuses a stale ring signature


def _worker_init(prior_path, threads):
    _quiet(threads)
    torch.set_default_dtype(torch.float64)
    _WORKER['prior'] = _load_prior(prior_path)


def _run_job(job, prior):
    try:
        return job['identifier'], compute_entry(prior=prior, **job), None
    except Exception as exc:                             # noqa: BLE001 - returned as a failure
        return job['identifier'], None, f'{type(exc).__name__}: {str(exc)[:300]}'


def _worker_run(job):
    return _run_job(job, _WORKER.get('prior'))


def _atomic_save(obj, path: Path):
    """torch.save through a temp file and os.replace. Raises, never swallows: a table or a
    part that silently failed to write would be recomputed or, worse, read stale."""
    tmp = path.with_name(path.name + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _part_path(parts: Path, identifier: str) -> Path:
    return parts / (hashlib.blake2b(identifier.encode(), digest_size=16).hexdigest() + '.pt')


def _resume_key(stamp: dict) -> str:
    keep = {k: stamp[k] for k in ('format', 'conditions', 'energy', 'search', 'basin',
                                  'internal_prior')}
    # contents, not paths: the same conditions file or prior moved to another directory
    # resumes; a different one with the same name does not
    keep['conditions'] = {'sha256': stamp['conditions']['sha256']}
    keep['internal_prior'] = (stamp['internal_prior'] or {}).get('sha256')
    # a DATABASE floor is another table: its identity joins the key. A search floor adds
    # nothing, so a parts directory written before floors had a source still resumes
    fl = stamp.get('floor') or {}
    if fl.get('source') == 'database':
        keep['floor'] = {'source': 'database', 'tol': fl['rescore_tol_kcal'],
                         **{k: fl['database'][k] for k in ('run_hash', 'header_hashes_sha256')}}
    return hashlib.sha256(json.dumps(keep, sort_keys=True, default=str).encode()).hexdigest()


def build_references(conditions_path, out_path, kwargs: Mapping, *,
                     search: Optional[Mapping] = None, internal_prior_path=None,
                     workers: int = 0, threads: int = 1, molecule_timeout: float = 900.0,
                     max_modes: int = BASIN_MAX_MODES, identifiers=None,
                     keep_parts: bool = False, database=None,
                     database_tol: Optional[float] = None, log=print) -> dict:
    """Build (or resume) the table. Returns ``{path, stamp, n_ok, failures, stalled, entries}``.

    ``workers=0`` computes in this process (tests); otherwise a spawn pool of that many
    processes, ``threads`` torch threads each. ``identifiers`` restricts the build to a
    subset (the stamp still names the whole conditions file).

    ``database`` (a build_conformer_database.py output directory) takes each condition's
    FLOOR from the database instead of the search (``database_floor``), within
    ``database_tol`` kcal/mol of the stored energy; the search parameters are then unused
    and the stamp's ``search`` is None. A condition the database holds no usable row for is
    a recorded failure. The stamp's ``floor`` names the source either way, and each entry's
    ``floor_source``.
    """
    search = dict(DEFAULT_SEARCH if search is None else search)
    missing = {'n_uniform', 'n_prior', 'n_seeds', 'steps'} - set(search)
    if missing:
        raise ValueError(f'search parameters missing: {sorted(missing)}')
    conditions_path, out_path = Path(conditions_path), Path(out_path)
    kwargs = member_kwargs(kwargs)
    mols = read_conditions(conditions_path)
    ids_all = sorted(m['identifier'] for m in mols)
    if identifiers is not None:
        want = set(identifiers)
        unknown = want - {m['identifier'] for m in mols}
        if unknown:
            raise ValueError(f'{len(unknown)} identifier(s) not in {conditions_path}, e.g. '
                             f'{sorted(unknown)[0]!r}')
        mols = [m for m in mols if m['identifier'] in want]
    db_recs, floor = None, {'source': 'search'}
    if database is not None:
        from build_conformer_database import read_conditions as read_database
        from build_conformer_database import refuse_other_member_kwargs
        if database_tol is None:
            raise ValueError('a database floor needs database_tol')
        db_info, db_recs = read_database(database, identifiers=[m['identifier'] for m in mols],
                                         log=log)
        refuse_other_member_kwargs(db_info, kwargs, 'reference build')
        # only the lowest row travels to a worker
        for r in db_recs.values():
            r['basins'] = {k: v[:1] for k, v in r['basins'].items()}
        floor = {'source': 'database', 'rescore_tol_kcal': float(database_tol),
                 'database': {k: db_info[k] for k in ('path', 'format', 'run_hash',
                                                      'header_hashes_sha256', 'n_shards',
                                                      'created_utc', 'git', 'window_kt')},
                 'rule': 'the lowest stored row of the condition, measured into the member '
                         '(build_conformer_database.match_rows) and re-scored; e_min is the '
                         're-score'}
        search = {k: 0 for k in search}                  # no search is run: no prior needed
    if search['n_prior'] > 0:
        if internal_prior_path is None:
            raise ValueError('n_prior > 0 needs internal_prior_path (or n_prior 0, stamped)')
        # LOADED HERE FIRST, not only in the workers: a pool initializer that raises is
        # respawned without end, so a stale prior would surface only as a stalled pool
        prior = _load_prior(internal_prior_path)
    else:
        internal_prior_path, prior = None, None
    stamp = {
        'format': FORMAT,
        # the identifiers themselves, not a hash of them: load_references refuses a table
        # that lacks an entry for any of them, and names which
        'conditions': {'path': str(conditions_path), 'sha256': file_sha256(conditions_path),
                       'n_identifiers': len(ids_all), 'identifiers': ids_all},
        'energy': defining_energy(kwargs),
        'energy_kwargs': _plain(kwargs),
        'level': kwargs.get('level'), 'force_field': kwargs.get('force_field', 'reference'),
        'energy_clip': kwargs.get('energy_clip'),
        'search': None if db_recs is not None else {
                   **{k: int(v) for k, v in search.items()}, 'optimizer': 'rprop',
                   'seed_rule': 'blake2b(salt + identifier) mod 2^31-1 = s; uniform draws at '
                                's, prior draws at s + 1; ETKDG seeds seed+1 .. seed+n_seeds',
                   'stereo': 'pinned at the elements the condition SMILES specifies, at the '
                             'reference conformer\'s configuration (legacy RDKit 3D '
                             'perception); elements it leaves open are free'},
        'floor': floor,
        'basin': {'max_modes': int(max_modes), 'accessible_kt': 10.0},
        'internal_prior': (None if internal_prior_path is None else
                           {'path': str(internal_prior_path),
                            'sha256': file_sha256(internal_prior_path)}),
        'currency': 'e_min: raw potential at T = 1, kcal/mol (bake_energies)',
        'dtype': 'float64',
        'code': code_revisions(),
        'created': datetime.datetime.now().isoformat(timespec='seconds'),
    }
    key = _resume_key(stamp)

    parts = out_path.with_name(out_path.name + '.parts')
    parts.mkdir(parents=True, exist_ok=True)
    done: Dict[str, dict] = {}
    foreign = 0
    for p in sorted(parts.glob('*.pt')):
        blob = torch.load(p, weights_only=False, map_location='cpu')
        if blob.get('key') != key:
            foreign += 1
            continue
        done[blob['identifier']] = blob['entry']
    if foreign:
        raise ValueError(f'{parts} holds {foreign} part(s) built under a different stamp '
                         f'(conditions, energy, search, basin or prior differ); remove the '
                         f'directory or choose another output path')
    todo = [m for m in mols if m['identifier'] not in done]
    log(f'references: {len(mols)} molecule(s) from {conditions_path}; {len(done)} resumed '
        f'from {parts}, {len(todo)} to compute on '
        f'{"this process" if workers <= 0 else f"{workers} worker(s)"}')

    jobs = [dict(identifier=m['identifier'], smiles=m['smiles'], z=m['z'], pos=m['pos'],
                 kwargs=kwargs, search=search, max_modes=max_modes,
                 **({} if db_recs is None else
                    dict(floor='database', db_rec=db_recs.get(m['identifier']),
                         db_tol=float(database_tol))))
            for m in todo]
    failures: Dict[str, str] = {}
    stalled = False
    t0 = time.time()

    def _take(ident, entry, err):
        if err is not None:
            failures[ident] = err
            log(f'  FAILED {ident}: {err}')
            return
        _atomic_save({'key': key, 'identifier': ident, 'entry': entry},
                      _part_path(parts, ident))
        done[ident] = entry
        n = len(done)
        if n % 25 == 0 or n == len(mols):
            log(f'  {n}/{len(mols)} done, {time.time() - t0:.0f} s')

    if workers <= 0:
        for job in jobs:
            _take(*_run_job(job, prior))
    elif jobs:
        import multiprocessing as mp

        pool = mp.get_context('spawn').Pool(
            int(workers), initializer=_worker_init,
            initargs=(internal_prior_path, int(threads)))
        try:
            it = pool.imap_unordered(_worker_run, jobs, chunksize=1)
            while True:
                try:
                    _take(*it.next(timeout=float(molecule_timeout)))
                except StopIteration:
                    break
                except mp.TimeoutError:
                    # THE WAIT IS BOUNDED: no molecule finished in the window, so a worker is
                    # stuck. Pool tasks cannot be cancelled one by one; stop them all and keep
                    # everything already written.
                    stalled = True
                    break
        finally:
            pool.terminate()
            pool.join()
        if stalled:
            for m in todo:
                if m['identifier'] not in done and m['identifier'] not in failures:
                    failures[m['identifier']] = (f'not completed: no molecule finished within '
                                                 f'{molecule_timeout:.0f} s')

    order = [m['identifier'] for m in mols]
    entries = {i: done[i] for i in order if i in done}
    table = {'stamp': stamp, 'entries': entries, 'failures': failures}
    _atomic_save(table, out_path)
    # the parts go only once the table covers the WHOLE file: a subset build's parts are
    # what a later full build resumes from
    if not failures and not keep_parts and len(entries) == len(ids_all):
        shutil.rmtree(parts, ignore_errors=True)
    fallbacks = _prior_fallbacks(entries)
    log(f'references: wrote {out_path} -- {len(entries)} of {len(ids_all)} identifier(s), '
        f'{len(failures)} failure(s){" (pool STALLED)" if stalled else ""}'
        + (f'; {len(fallbacks)} molecule(s) fell back to uniform starts because the prior '
           f'draw raised' if fallbacks else ''))
    return {'path': out_path, 'stamp': stamp, 'n_ok': len(entries), 'failures': failures,
            'stalled': stalled, 'entries': entries, 'prior_fallbacks': fallbacks}


def _prior_fallbacks(entries: Mapping[str, dict]) -> List[str]:
    return [i for i, e in entries.items() if 'prior_error' in e['starts']]


# ------------------------------------------------------------------------------ load


class ReferenceTable:
    """A loaded, checked table: ``table[identifier]`` is one entry dict.

    ``missing`` lists the conditions file's identifiers without an entry (empty unless the
    table was loaded with ``require_complete=False``), ``failures`` the recorded reasons.
    """

    def __init__(self, stamp: dict, entries: Dict[str, dict], failures: Dict[str, str],
                 missing: Sequence[str] = ()):
        self.stamp, self.entries, self.failures = stamp, entries, failures
        self.missing = list(missing)

    @property
    def prior_fallbacks(self) -> List[str]:
        """Identifiers whose prior draw raised, so their prior starts were replaced by uniform
        ones -- a search the depth calibration did not measure (``starts['prior_error']``)."""
        return _prior_fallbacks(self.entries)

    def __len__(self):
        return len(self.entries)

    def __contains__(self, identifier):
        return identifier in self.entries

    def __getitem__(self, identifier):
        return self.entries[identifier]

    @property
    def identifiers(self) -> List[str]:
        return list(self.entries)

    def eval_refs(self, identifiers=None) -> Dict[str, dict]:
        """``per_molecule_block``'s ``refs``: ``{identifier: {e_min, basin_ref, target_tc}}``.

        ``e_min`` is the CONDITION's floor: pinned at the elements its SMILES specifies, and
        over every stereoisomer for a stereo-free SMILES (see the module docstring).
        Identifiers without an entry (possible only in a table loaded with
        ``require_complete=False``) are left out, which makes that molecule's target
        readings abstain with ``*_available = 0``.
        """
        ids = self.identifiers if identifiers is None else list(identifiers)
        return {i: {'e_min': self.entries[i]['e_min'], 'basin_ref': self.entries[i]['basin_ref'],
                    'target_tc': self.entries[i]['target_tc']}
                for i in ids if i in self.entries}

    def column(self, name: str, identifiers: Sequence[str]) -> np.ndarray:
        """One scalar field in the given order, e.g. ``e_min`` gathered by a mol_id registry.
        A missing identifier or a None value raises: a NaN here would read as a value (a
        floor, or ``n_accessible``, which is None where the basin enumeration was skipped)."""
        missing = [i for i in identifiers if i not in self.entries]
        if missing:
            raise KeyError(f'{len(missing)} identifier(s) have no reference entry, e.g. '
                           f'{missing[0]!r}')
        vals = [self.entries[i][name] for i in identifiers]
        none = [i for i, v in zip(identifiers, vals) if v is None]
        if none:
            raise ValueError(f'{name!r} is None for {len(none)} identifier(s), e.g. '
                             f'{none[0]!r}; select the entries that carry it first')
        return np.asarray(vals, dtype=np.float64)

    def verify(self, members: Mapping, tol: float = EMIN_RESCORE_TOL,
               allow_missing: bool = False) -> float:
        """Re-score each ``e_min_state`` through the RUN's member; refuse a gap above ``tol``.

        ``members`` is ``{identifier: ConformerTorsions}`` (e.g.
        ``MultiConformerTorsions._members``). A member with no entry is refused unless
        ``allow_missing`` (the partner of ``require_complete=False``), where it is skipped.
        Returns the worst |dE| in kcal/mol.
        """
        from energies.conformer_data import bake_energies

        worst, bad = 0.0, []
        for ident, member in members.items():
            e = self.entries.get(ident)
            if e is None:
                if not allow_missing:
                    bad.append(f'{ident}: no entry in the table')
                continue
            if int(member.ndim) != int(e['k']):
                bad.append(f'{ident}: member k {int(member.ndim)} against the table\'s {e["k"]}')
                continue
            with torch.no_grad():
                u = float(bake_energies(member, e['e_min_state'].reshape(1, -1))[0])
            gap = abs(u - float(e['e_min']))
            if not gap <= tol:
                bad.append(f'{ident}: e_min_state re-scores at {u:.6g} against the stored '
                           f'{float(e["e_min"]):.6g} kcal/mol')
            worst = max(worst, gap) if np.isfinite(gap) else float('inf')
        if bad:
            raise StaleReferencesError(
                f'{len(bad)} reference entr{"y" if len(bad) == 1 else "ies"} do not re-score '
                f'through the run\'s members (tol {tol:g} kcal/mol):\n  ' + '\n  '.join(bad[:10]))
        return worst


def load_references(path, *, conditions_path, energy_kwargs,
                    require_complete: bool = True) -> ReferenceTable:
    """Load a table and REFUSE it unless it was built for this conditions file and energy.

    ``conditions_path`` is the file the run reads (its sha256 must equal the stamp's);
    ``energy_kwargs`` the ConformerTorsions arguments the run builds its members with (the
    member-defining ones, resolved against the signature defaults, must equal the stamp's).
    Both are required: a table checked against nothing is the silent case this exists to
    prevent. A table without an entry for every identifier of the file (a failed or stalled
    molecule, or a subset build) raises ``IncompleteReferencesError`` unless
    ``require_complete=False``, which loads it with ``missing`` listing the gaps. Call
    ``verify`` on the live members for the re-score check.
    """
    blob = torch.load(path, weights_only=False, map_location='cpu')
    stamp = blob.get('stamp', {}) if isinstance(blob, dict) else {}
    if stamp.get('format') != FORMAT:
        raise StaleReferencesError(f'{path}: format {stamp.get("format")!r}, this reader '
                                   f'takes {FORMAT!r}')
    problems = []
    sha = file_sha256(conditions_path)
    if sha != stamp['conditions']['sha256']:
        problems.append(f'conditions file {conditions_path} has sha256 {sha[:16]}..., the '
                        f'table was built from {stamp["conditions"]["path"]} with '
                        f'{stamp["conditions"]["sha256"][:16]}...')
    want = defining_energy(member_kwargs(energy_kwargs))
    # AN ARGUMENT THE STAMP DOES NOT NAME TAKES ITS CODE DEFAULT, the rule `defining_energy`
    # applies to a config: a table stamped before an argument existed was built under what is
    # now that argument's default (`double_bond_box_deg`, None), not under "absent".
    have = dict(stamp['energy'])
    for name, p in _ct_parameters().items():
        if name in want and name not in have and p.default is not inspect.Parameter.empty:
            have[name] = _plain(p.default)
    for k in sorted(set(want) | set(have)):
        if want.get(k, '<absent>') != have.get(k, '<absent>'):
            problems.append(f'energy {k}: run has {want.get(k, "<absent>")!r}, table has '
                            f'{have.get(k, "<absent>")!r}')
    if problems:
        raise StaleReferencesError(f'{path} does not describe this run:\n  '
                                   + '\n  '.join(problems))
    entries, failures = blob['entries'], blob.get('failures', {})
    missing = [i for i in stamp['conditions']['identifiers'] if i not in entries]
    if missing and require_complete:
        why = failures.get(missing[0], 'not built (a subset build)')
        raise IncompleteReferencesError(
            f'{path} has no entry for {len(missing)} of {len(stamp["conditions"]["identifiers"])} '
            f'identifier(s) ({len(failures)} recorded failure(s)), e.g. {missing[0]!r}: {why}. '
            f'Rerun the build to resume, or load with require_complete=False, under which '
            f'those molecules\' target readings abstain')
    return ReferenceTable(stamp, entries, failures, missing)


# ------------------------------------------------------------------------------- CLI


def cost_table(entries: Mapping[str, dict]) -> str:
    """Per-molecule wall time by k band, labelled and captioned (AGENTS.md tables)."""
    rows = [(int(e['k']), e['cost_s']) for e in entries.values()]
    if not rows:
        return '(no entries)'
    bands = [(0, 30), (31, 45), (46, 60), (61, 10 ** 6)]
    lines = ['| k band (coordinates) | molecules | total p50 (s) | total p90 (s) | '
             'total max (s) | descend p50 (s) | starts p50 (s) | basin p50 (s) |',
             '|---|---|---|---|---|---|---|---|']
    for lo, hi in bands + [(0, 10 ** 6)]:
        sel = [c for k, c in rows if lo <= k <= hi]
        if not sel:
            continue
        q = lambda key, p: float(np.percentile([c[key] for c in sel], p))
        name = 'all' if (lo, hi) == (0, 10 ** 6) else (f'{lo}-{hi}' if hi < 10 ** 6
                                                      else f'>= {lo}')
        lines.append(f'| {name} | {len(sel)} | {q("total", 50):.2f} | {q("total", 90):.2f} | '
                     f'{q("total", 100):.2f} | {q("descend", 50):.2f} | {q("starts", 50):.2f} | '
                     f'{q("basin", 50):.2f} |')
    return '\n'.join(lines)


def _energy_config_from_yaml(path):
    """energy_config after the loader's own resolution -- what the trainer would read."""
    import utils
    args = utils.resolve_derived_config(
        utils.preflight_config(utils.dict2namespace(utils.load_yaml(str(path)))))
    return vars(args.energy_config)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--conditions', required=True, help='conditions file (molecules_path form)')
    ap.add_argument('--config', required=True,
                    help='run config whose energy_config builds the members')
    ap.add_argument('--out', default=None,
                    help='table path (default: <conditions stem>.references.pt beside it)')
    ap.add_argument('--internal-prior', default=None,
                    help='fitted InternalPrior for the prior starts (default: '
                         'energy_config.internal_prior_path)')
    for name, v in DEFAULT_SEARCH.items():
        ap.add_argument('--' + name.replace('_', '-'), type=int, default=v)
    ap.add_argument('--max-modes', type=int, default=BASIN_MAX_MODES)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--threads', type=int, default=1, help='torch threads per worker')
    ap.add_argument('--molecule-timeout', type=float, default=900.0,
                    help='seconds without any molecule finishing before the pool is stopped')
    ap.add_argument('--keep-parts', action='store_true')
    ap.add_argument('--database', default=None,
                    help='take each e_min FLOOR from this conformer database '
                         '(build_conformer_database.py output directory) instead of the '
                         'search; basin_ref is computed as without it')
    ap.add_argument('--database-rescore-tol', type=float, default=None,
                    help='kcal/mol; a database floor row re-scoring further from its stored '
                         'energy is a failure (default build_conformer_set.'
                         'DATABASE_RESCORE_TOL)')
    args = ap.parse_args(argv)

    _quiet(args.threads)
    ec = _energy_config_from_yaml(args.config)
    kwargs = member_kwargs(ec)
    search = {k: getattr(args, k) for k in DEFAULT_SEARCH}
    prior = args.internal_prior or ec.get('internal_prior_path')
    db_tol = args.database_rescore_tol
    if args.database is not None:
        if db_tol is None:
            from build_conformer_set import DATABASE_RESCORE_TOL
            db_tol = DATABASE_RESCORE_TOL
        prior = None                       # no search, so no prior starts
    elif search['n_prior'] > 0:
        if not prior or not Path(prior).exists():
            raise SystemExit(f'REFUSING: n_prior {search["n_prior"]} needs the fitted '
                             f'InternalPrior, and {prior!r} does not exist. Pass '
                             f'--internal-prior, or --n-prior 0 to build without prior '
                             f'starts (stamped).')
    else:
        prior = None
    cond = Path(args.conditions)
    out = Path(args.out) if args.out else cond.with_name(cond.stem + '.references.pt')
    t0 = time.time()
    res = build_references(cond, out, kwargs, search=search, internal_prior_path=prior,
                           workers=args.workers, threads=args.threads,
                           molecule_timeout=args.molecule_timeout, max_modes=args.max_modes,
                           keep_parts=args.keep_parts, database=args.database,
                           database_tol=db_tol)
    ents = res['entries']
    how = (f'floor from the database {args.database} (descend = measuring its lowest row)'
           if args.database is not None else
           f'{sum(search[k] for k in ("n_uniform", "n_prior", "n_seeds")) + 1} starts x '
           f'{search["steps"]} steps')
    print()
    print(f'Per-molecule wall time of this build, by molecule size k (level '
          f'{kwargs.get("level")!r}, force field {kwargs.get("force_field", "reference")!r}; '
          f'{args.workers} worker(s) x {args.threads} thread(s); {how}). Resumed molecules '
          f'keep the cost of the run that computed them.')
    print(cost_table(ents))
    if ents:
        tot = sum(e['cost_s']['total'] for e in ents.values())
        pins = {c: sum(1 for e in ents.values() if e['stereo_pin'] == c)
                for c in ('none', 'partial', 'full')}
        print(f'\n{len(ents)} molecule(s), {tot / 3600:.3f} CPU-hours of per-molecule work, '
              f'{time.time() - t0:.0f} s wall this invocation.')
        print(f'Stereo pin from the condition SMILES: {pins["full"]} full, {pins["partial"]} '
              f'partial (elements left open, of which =NH imines: '
              f'{sum(e["stereo_open"]["imine_nh"] for e in ents.values() if e["stereo_pin"] == "partial")}), '
              f'{pins["none"]} none (floor over every stereoisomer). Molecules with a candidate '
              f'below the floor that breaks the pin: '
              f'{sum(1 for e in ents.values() if (e["n_below_other_stereo"] or 0) > 0)}'
              f' (not recorded for a database floor).')
        if res['prior_fallbacks']:
            print(f'{len(res["prior_fallbacks"])} molecule(s) fell back to uniform starts '
                  f'because the prior draw raised (entry starts.prior_error), e.g. '
                  f'{res["prior_fallbacks"][0]!r}.')
    if res['failures']:
        print(f'\n{len(res["failures"])} molecule(s) FAILED or unfinished; a rerun resumes the '
              f'finished ones:')
        for i, err in list(res['failures'].items())[:20]:
            print(f'  {i}: {err}')
        return 3 if res['stalled'] else 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
