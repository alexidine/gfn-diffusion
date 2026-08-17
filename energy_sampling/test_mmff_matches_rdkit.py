"""``ff_from_mmff`` must reproduce RDKit's MMFF94 TERM BY TERM, not just in total.

A total-energy check is not enough here. Two of the conversions were wrong on the first
pass in ways a total would have hidden or misattributed:

* vdW was reading GetMMFFVdWParams' first two returns instead of its donor-acceptor
  rescaled third and fourth. Every molecule without a hydrogen bond still matched to
  1e-14, so only a per-term check on an H-bonding molecule catches it.
* The torsion sign convention is invisible at phi = 0, where the V2 term vanishes under
  either sign -- which is exactly where an aromatic ring sits. Hence the perturbation
  below, and hence the aliphatic molecules in the list.

Geometries are PERTURBED away from the MMFF minimum so that no term can pass by being
near zero, and the perturbation is seeded so a failure is reproducible.
"""
import numpy as np
import pytest
import torch

torch.manual_seed(0)

rdkit = pytest.importorskip('rdkit')
from rdkit import Chem                                          # noqa: E402
from rdkit.Chem import AllChem                                  # noqa: E402

from mxtaltools.conformers.builder import collate                # noqa: E402
from mxtaltools.conformers.topology import spec_from_mol         # noqa: E402
from mxtaltools.conformers.energy import (ff_from_mmff,          # noqa: E402
                                          intramolecular_energy)

RDKIT_TERMS = ('Bond', 'Angle', 'StretchBend', 'Oop', 'Torsion', 'VdW', 'Ele')

# term name in our components dict -> RDKit's term name
TERM_MAP = {'bond': 'Bond', 'angle': 'Angle', 'stretch_bend': 'StretchBend',
            'oop': 'Oop', 'torsion': 'Torsion', 'lj': 'VdW', 'electrostatic': 'Ele'}

# Chosen so that between them every term is exercised AWAY from zero:
#   amide      -- H-bond donor/acceptor (the vdW rescaling), big electrostatics, sp2 oop
#   phenol     -- aromatic torsions and oop
#   branched   -- pure aliphatic torsions, no electrostatics at all
#   cyclohexane-- ring closure bonds
#   nitrile    -- an MMFF LINEAR centre, which uses a different angle functional form
MOLECULES = ['CC(=O)NC', 'c1ccccc1O', 'CCC(C)CO', 'C1CCCCC1', 'CC#N', 'CC(=O)Nc1ccccc1']


def rdkit_term_energy(mol, term):
    """RDKit's energy with every term but `term` switched off."""
    props = AllChem.MMFFGetMoleculeProperties(mol)
    for u in RDKIT_TERMS:
        getattr(props, f'SetMMFF{u}Term')(u == term)
    return AllChem.MMFFGetMoleculeForceField(mol, props).CalcEnergy()


def perturbed_mol(smi, sigma=0.06, seed=0xbeef):
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    assert AllChem.EmbedMolecule(mol, randomSeed=seed) == 0, f'embed failed for {smi}'
    AllChem.MMFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    rng = np.random.default_rng(seed)
    pos = conf.GetPositions() + rng.normal(0, sigma, (mol.GetNumAtoms(), 3))
    for a in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(a, pos[a].tolist())
    return mol, pos


def build_ff(mol, pos):
    """A one-molecule BatchedTree plus its MMFF force field, in placement-slot order."""
    spec = spec_from_mol(mol)
    tree = collate([spec], device='cpu')
    ff = ff_from_mmff(tree, mol, spec.perm, dtype=torch.float64)
    # positions must be reordered into placement-slot order to match the index arrays
    xyz = torch.as_tensor(pos[np.asarray(spec.perm)], dtype=torch.float64)
    return tree, ff, xyz


@pytest.mark.parametrize('smi', MOLECULES)
def test_every_term_matches_rdkit(smi):
    mol, pos = perturbed_mol(smi)
    tree, ff, xyz = build_ff(mol, pos)
    _, ours = intramolecular_energy(tree, xyz, ff, components=True)

    failures = []
    for key, rd_name in TERM_MAP.items():
        ref = rdkit_term_energy(mol, rd_name)
        got = float(ours[key].sum())
        # 3e-4 is the precision of what RDKit's accessors REPORT, not of the functional
        # form: theta0 comes back at three decimals and koop at two or three. For each of
        # these three terms the observed residual sits BELOW the noise that rounding
        # alone predicts (measured: angle 3e-4 vs 4e-4 predicted, oop 1e-5 vs 3e-3), and
        # it scales with molecule size the way rounding does. The other four terms carry
        # no such rounding and land at 1e-12.
        tol = 3e-4 if rd_name in ('Angle', 'StretchBend', 'Oop') else 1e-9
        if abs(got - ref) > tol:
            failures.append(f'{rd_name}: ours={got:.9f} rdkit={ref:.9f} '
                            f'err={abs(got - ref):.2e} tol={tol:.0e}')
    assert not failures, f'{smi}\n  ' + '\n  '.join(failures)


@pytest.mark.parametrize('smi', MOLECULES)
def test_total_matches_rdkit(smi):
    mol, pos = perturbed_mol(smi)
    tree, ff, xyz = build_ff(mol, pos)
    total = float(intramolecular_energy(tree, xyz, ff).sum())
    ref = AllChem.MMFFGetMoleculeForceField(
        mol, AllChem.MMFFGetMoleculeProperties(mol)).CalcEnergy()
    assert abs(total - ref) < 1e-3, f'{smi}: ours={total:.6f} rdkit={ref:.6f}'


def test_vdw_donor_acceptor_rescaling_is_load_bearing():
    """Using the un-rescaled R*/eps must FAIL, or the test above proves nothing.

    Re-introduces the exact bug the per-term check caught: reading GetMMFFVdWParams'
    first two returns rather than its donor-acceptor rescaled last two.
    """
    mol, pos = perturbed_mol('CC(=O)NC')
    tree, ff, xyz = build_ff(mol, pos)
    props = AllChem.MMFFGetMoleculeProperties(mol)
    spec = spec_from_mol(mol)
    perm = np.asarray(spec.perm)

    n_at = mol.GetNumAtoms()
    pi = ff.pair_index.cpu().numpy() % n_at
    raw = torch.as_tensor(
        np.asarray([props.GetMMFFVdWParams(int(perm[int(a)]), int(perm[int(b)]))[0]
                    for a, b in pi]), dtype=torch.float64)
    raw_eps = torch.as_tensor(
        np.asarray([props.GetMMFFVdWParams(int(perm[int(a)]), int(perm[int(b)]))[1]
                    for a, b in pi]), dtype=torch.float64)

    ref = rdkit_term_energy(mol, 'VdW')
    good = float(intramolecular_energy(tree, xyz, ff, components=True)[1]['lj'].sum())

    import dataclasses
    bad_ff = dataclasses.replace(ff, vdw_rstar=raw, epsilon=raw_eps)
    bad = float(intramolecular_energy(tree, xyz, bad_ff, components=True)[1]['lj'].sum())

    assert abs(good - ref) < 1e-9, f'rescaled params should match: {good} vs {ref}'
    assert abs(bad - ref) > 1e-3, (
        'un-rescaled R*/eps produced the same answer, so this molecule does not actually '
        f'exercise the donor-acceptor path: bad={bad} ref={ref}')


def test_linear_angle_form_is_load_bearing():
    """A linear centre must use 143.9325 ka (1 + cos theta), not the cubic bend."""
    mol, pos = perturbed_mol('CC#N')
    tree, ff, xyz = build_ff(mol, pos)
    assert bool(ff.angle_linear.any()), 'CC#N should have an MMFF linear centre'

    import dataclasses
    ref = rdkit_term_energy(mol, 'Angle')
    good = float(intramolecular_energy(tree, xyz, ff, components=True)[1]['angle'].sum())
    bent = dataclasses.replace(ff, angle_linear=torch.zeros_like(ff.angle_linear))
    bad = float(intramolecular_energy(tree, xyz, bent,
                                      components=True)[1]['angle'].sum())

    assert abs(good - ref) < 3e-4
    assert abs(bad - ref) > 1e-3, (
        f'treating the linear centre as bent gave the same answer: bad={bad} ref={ref}')


def test_softcore_leaves_mmff_untouched_above_the_switch():
    """vdw_softcore_frac must not perturb any thermally reachable geometry."""
    mol, pos = perturbed_mol('CCC(C)CO')
    spec = spec_from_mol(mol)
    tree = collate([spec], device='cpu')
    xyz = torch.as_tensor(pos[np.asarray(spec.perm)], dtype=torch.float64)

    exact = ff_from_mmff(tree, mol, spec.perm, dtype=torch.float64)
    soft = ff_from_mmff(tree, mol, spec.perm, dtype=torch.float64,
                        vdw_softcore_frac=0.3)
    e_exact = float(intramolecular_energy(tree, xyz, exact).sum())
    e_soft = float(intramolecular_energy(tree, xyz, soft).sum())
    assert abs(e_exact - e_soft) < 1e-9, (
        f'softening changed a normal geometry: {e_exact} vs {e_soft}')


def test_softcore_bounds_a_hard_clash():
    """With softening on, a fully overlapped pair stays finite with a BOUNDED gradient.

    The bounded gradient is the point, not merely a smaller number: an optimiser or
    sampler that lands on a clash has to be able to walk back out of it.
    """
    from mxtaltools.conformers.energy import buffered_147
    r = torch.tensor([1e-6, 0.05, 0.2, 0.5, 1.0, 2.0, 3.5, 5.0], dtype=torch.float64)
    R = torch.full_like(r, 3.5)
    eps = torch.full_like(r, 0.05)
    frac = 0.3

    hard = buffered_147(r, R, eps, 0.0)
    soft = buffered_147(r, R, eps, frac)

    assert torch.isfinite(soft).all(), 'softened form produced a non-finite energy'
    assert soft[0] < hard[0] / 1e3, (
        f'softening barely helped at r->0: {float(soft[0]):.3e} vs {float(hard[0]):.3e}')

    above = r >= frac * 3.5
    assert torch.allclose(hard[above], soft[above], atol=1e-12), \
        'softening leaked above the switch radius'

    # gradient below the switch must be constant, and vastly smaller than 14-7's
    rg = torch.linspace(1e-4, frac * 3.5 * 0.99, 200, dtype=torch.float64)
    Rg, eg = torch.full_like(rg, 3.5), torch.full_like(rg, 0.05)
    rg.requires_grad_(True)
    buffered_147(rg, Rg, eg, frac).sum().backward()
    g = rg.grad
    assert (g.max() - g.min()).abs() < 1e-9, 'linear continuation is not linear'

    rh = torch.tensor([0.2], dtype=torch.float64, requires_grad=True)
    buffered_147(rh, torch.tensor([3.5], dtype=torch.float64),
                 torch.tensor([0.05], dtype=torch.float64), 0.0).sum().backward()
    assert g.abs().max() < rh.grad.abs().item() / 1e3, (
        f'softened gradient {float(g.abs().max()):.3e} is not far below the '
        f'unsoftened {float(rh.grad.abs()):.3e}')


def test_softcore_is_continuous_at_the_switch():
    """No step at r = f R*, or the sampler sees a discontinuous reward.

    A fixed absolute tolerance cannot express this: the switch sits on a wall whose slope
    is ~3.4e3 kcal/mol/A, so even a perfectly continuous join shows a 7e-4 difference
    across a 2e-7 gap. What separates a step from a steep slope is that the gap SHRINKS
    WITH THE INTERVAL -- linearly for continuous, not at all for a step.
    """
    from mxtaltools.conformers.energy import buffered_147
    frac, R, eps = 0.3, 3.5, 0.05
    args = (torch.tensor([R], dtype=torch.float64),
            torch.tensor([eps], dtype=torch.float64), frac)

    def jump(d):
        lo = torch.tensor([frac * R - d], dtype=torch.float64)
        hi = torch.tensor([frac * R + d], dtype=torch.float64)
        return abs(float(buffered_147(lo, *args)) - float(buffered_147(hi, *args)))

    j1, j2 = jump(1e-6), jump(1e-7)
    assert j2 < j1 / 5, (
        f'jump did not shrink with the interval ({j1:.2e} -> {j2:.2e}), which means a '
        f'genuine step at the switch rather than a steep but continuous join')
