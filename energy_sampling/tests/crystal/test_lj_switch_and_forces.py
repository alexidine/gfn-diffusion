"""The LJ cutoff switch as the trainer's default (owner 2026-11-19) and stored forces (build_noised_prior --forces).

What each test would catch:
  * an LJ-family energy run on the hard cutoff when analyze_kwargs does not set lj_envelope, or an explicit null not
    honoured (the hard cutoff stays available);
  * a prior file's energy identity that does not carry the effective switch: an unswitched (old) file loading under the
    switched default, or a refusal that does not name the restamp tool;
  * the prior scan cache keyed on the raw config only, so it would reuse hard-cutoff scores;
  * the switch entering the checkpoint problem definition (it renames every checkpoint: resumes would start fresh);
  * a stored force that is not the gradient of the stored mol_energy at the row's FD chart point.

    python -m pytest -q tests/crystal/test_lj_switch_and_forces.py
"""
import argparse
import types
from pathlib import Path

import pytest
import torch

import noised_prior as NP
import utils
from energies.molecular_crystal import LJ_ENVELOPE_A, MolecularCrystal, effective_analyze_kwargs
from mxtaltools.dataset_utils.data_classes import MolData
from mxtaltools.dataset_utils.utils import collate_data_list

pytestmark = pytest.mark.fast

CANONICAL = Path(__file__).resolve().parents[2] / 'configs' / 'mk_dev.yaml'


def _args(capsys):
    a = utils.get_train_args(['--config', str(CANONICAL)])
    capsys.readouterr()
    return a


def _energy(args, **over):
    cfg = {'device': 'cpu', 'energy_function': args.energy_function, 'mlip_path': args.mlip_path,
           'space_groups': args.space_groups, 'z_primes': args.z_primes, 'sg_conditioning': args.sg_conditioning,
           'temperature_conditioning': args.temperature_conditioning, 'zp_conditioning': args.zp_conditioning}
    cfg.update(vars(args.energy_config))
    cfg.update(over)
    return MolecularCrystal(**cfg)


def test_lj_family_energies_are_switched_unless_the_config_says_otherwise():
    assert effective_analyze_kwargs('elj', {}) == {'lj_envelope': LJ_ENVELOPE_A} and LJ_ENVELOPE_A == 1.0
    assert effective_analyze_kwargs('elj', {'lj_envelope': None}) == {'lj_envelope': None}   # the hard cutoff
    assert effective_analyze_kwargs('elj', {'lj_envelope': 0.5}) == {'lj_envelope': 0.5}
    assert effective_analyze_kwargs('uma', {}) == {}
    assert effective_analyze_kwargs('elj', argparse.Namespace(cutoff=8)) == {'cutoff': 8, 'lj_envelope': 1.0}


def test_an_unswitched_prior_refuses_under_the_switched_default_and_the_refusal_names_the_restamp(capsys):
    args = _args(capsys)
    ef = _energy(args)
    run = NP.identity(args, ef)
    assert run['analyze_kwargs']['lj_envelope'] == 1.0
    old = dict(run, analyze_kwargs={k: v for k, v in run['analyze_kwargs'].items() if k != 'lj_envelope'})
    bad = NP.identity_mismatch(old, run)
    assert bad and bad[0].startswith('analyze_kwargs')
    hint = NP.restamp_hint(old, run, 'prior.pt')
    assert 'restamp_prior.py prior.pt' in hint
    assert NP.restamp_hint(dict(old, lj_coeff=0.5), run, 'p') == ''      # anything else differs: rebuild, no hint
    # an explicit hard cutoff is the identity every file written before the switch carries
    hard = _energy(args, analyze_kwargs={'lj_envelope': None})
    assert NP.identity_mismatch(old, NP.identity(args, hard)) == []


def test_the_scan_cache_identity_carries_the_effective_switch(capsys):
    import prior_scan_cache as PSC
    args = _args(capsys)
    args.prior_path = str(CANONICAL)   # any existing file: the identity reads its size
    a = PSC.scan_identity(args, 10, 1.0)
    assert a['analyze_kwargs_effective'].get('lj_envelope') == 1.0
    args.energy_config.analyze_kwargs = {'lj_envelope': None}
    assert PSC.scan_identity(args, 10, 1.0)['analyze_kwargs_effective'].get('lj_envelope') is None


def test_the_switch_does_not_rename_checkpoints(capsys):
    args = _args(capsys)
    args.energy_config.analyze_kwargs = {}
    before = utils.problem_hash(utils.get_problem_definition(args))
    args.energy_config.analyze_kwargs = {'lj_envelope': 1.0}
    after_def = utils.get_problem_definition(args)
    assert utils.problem_hash(after_def) == before
    stored = dict(after_def, energy_config=dict(after_def['energy_config'], analyze_kwargs={'lj_envelope': 1.0}))
    assert utils.normalize_problem_def(stored) == utils.normalize_problem_def(after_def)


def _molecule(identifier, n_atoms, scale=1.6):
    base = torch.tensor([[-0.8, -0.2, 0.0], [0.7, -0.1, 0.1], [0.1, 0.8, -0.1], [0.0, 0.0, 0.9], [-0.5, 0.6, 0.5]])
    return MolData(z=torch.tensor([6, 7, 8, 6, 7][:n_atoms], dtype=torch.long), pos=base[:n_atoms] * scale,
                   x=torch.zeros((n_atoms, 1), dtype=torch.float32), identifier=identifier,
                   mol_volume=torch.tensor(18.0 * n_atoms * scale ** 3), mass=torch.tensor(13.0 * n_atoms),
                   radius=torch.tensor(1.2 * scale), z_prime=torch.tensor(1, dtype=torch.long))


def test_a_stored_force_is_the_gradient_of_the_stored_mol_energy_at_the_chart_point(capsys):
    from data_processing.build_noised_prior import score
    args = _args(capsys)
    ef = _energy(args, crystal_coordinates='fd')
    n = 6
    g = torch.Generator().manual_seed(5)
    s = torch.rand(n, ef.fd_layout.dim, generator=g) * 1.2 - 0.6
    s[:, :3] = 0.3 + 0.2 * torch.rand(n, 3, generator=g)   # cells long enough for a physical packing
    mols = collate_data_list([_molecule(f'f{i}', 3 + i % 3) for i in range(n)])
    with ef.allow_unreferenced():
        _, rows = ef.analyze_crystal_batch(s, mols, torch.full((n,), 2.5), return_batch=True)
    out, kept = score(ef, rows, 2.5, 'cpu', 4096, forces=True)
    assert bool(kept.all())
    b = out[0]
    assert tuple(b.force.shape) == (n, ef.fd_layout.dim) and b.force.dtype == torch.float32
    ok, x0, _ = NP.fd_chart_points(ef, b, device='cpu')
    x0 = x0.double()
    interior = (x0.abs() < 0.98)
    def en(x):
        with torch.no_grad(), ef.allow_unreferenced():
            _, c = ef.batched_analyze_crystal_batch(x.float(), b, torch.full((n,), 2.5), return_batch=True)
        return c.mol_energy.reshape(-1).double()
    torch.testing.assert_close(en(x0), b.mol_energy.reshape(-1).double(), rtol=0, atol=1e-3)
    checked = 0
    for j in range(x0.shape[1]):
        h = 2e-3
        xp, xm = x0.clone(), x0.clone()
        xp[:, j] += h
        xm[:, j] -= h
        cd = (en(xp) - en(xm)) / (2 * h)
        m = interior[:, j]
        err = (b.force[:, j].double() - cd).abs()[m]
        assert bool((err <= 0.05 * cd.abs()[m] + 1.0).all()), (j, b.force[:, j][m], cd[m])
        checked += int(m.sum())
    assert checked >= n * x0.shape[1] // 2


def test_merging_groups_pads_each_force_column_with_nan_to_the_widest_chart():
    from types import SimpleNamespace
    from data_processing.merge_noised_priors import pad_forces

    class Rows(SimpleNamespace):
        def keys(self):
            return list(vars(self))
    a = Rows(force=torch.arange(24.).reshape(2, 12))
    b = Rows(force=torch.arange(27.).reshape(3, 9))
    assert pad_forces([a, b]) == 12
    assert a.force.shape == (2, 12) and torch.equal(a.force, torch.arange(24.).reshape(2, 12))
    assert b.force.shape == (3, 12) and torch.equal(b.force[:, :9], torch.arange(27.).reshape(3, 9))
    assert bool(torch.isnan(b.force[:, 9:]).all())
    assert pad_forces([Rows(x=1), Rows(x=2)]) is None
    with pytest.raises(SystemExit):
        pad_forces([Rows(force=torch.zeros(1, 9)), Rows(x=1)])
