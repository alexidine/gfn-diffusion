"""noised_prior.py: the load-time selection of a pre-noised prior file (docs/design/prior_dataset_build.md)."""
import types

import pytest
import torch

import noised_prior as NP

pytestmark = pytest.mark.fast


def _batch(identifier, sg, zp, **tags):
    b = types.SimpleNamespace(identifier=list(identifier), sg_ind=torch.tensor(sg), z_prime=torch.tensor(zp))
    for k, v in tags.items():
        setattr(b, k, torch.tensor(v))
    return b


def _settings(**over):
    s = NP.settings(types.SimpleNamespace(noised_prior=types.SimpleNamespace(**over)))
    return s


def test_a_condition_is_identifier_space_group_and_z_prime():
    b = _batch(['a', 'a', 'b', 'a', 'a'], [2, 2, 2, 14, 2], [1, 1, 1, 1, 2])
    g = NP.condition_groups(b)
    assert g.tolist() == [0, 0, 1, 2, 3]


def test_the_floor_is_taken_over_eligible_rows_only():
    e = torch.tensor([5.0, 1.0, 3.0, 7.0], dtype=torch.float64)
    g = torch.tensor([0, 0, 1, 1])
    eligible = torch.tensor([True, False, True, True])
    assert NP.condition_floor(e, g, eligible).tolist() == [5.0, 5.0, 3.0, 3.0]


def test_selection_keeps_training_children_inside_both_windows():
    T = 2.0
    e = torch.tensor([0.0, 1.0, 13.0, 2.0, 0.5, 1.0], dtype=torch.float64)    # floor 0 (row 0, a training anchor)
    packing = torch.tensor([0.7, 0.7, 0.7, 0.99, 0.7, 0.7], dtype=torch.float64)
    kinds = torch.tensor([NP.KIND_ANCHOR, NP.KIND_CHILD, NP.KIND_CHILD, NP.KIND_CHILD, NP.KIND_ANCHOR, NP.KIND_CHILD])
    splits = torch.tensor([0, 0, 0, 0, 1, 1])
    groups = torch.zeros(6, dtype=torch.long)
    masks, floor = NP.select_rows(e, packing, kinds, splits, groups, T, _settings(window_kT=6.0))
    assert floor.tolist() == [0.0] * 6
    # row 1: a training child within 6 kT (12 energy units); row 2 beyond it; row 3 outside the packing window
    assert masks['train'].tolist() == [False, True, False, False, False, False]
    assert masks['heldout'].tolist() == [False, False, False, False, False, True]
    assert masks['anchors'].tolist() == [True, False, False, False, False, False]


def test_held_out_rows_never_set_the_floor():
    e = torch.tensor([10.0, -50.0, 11.0], dtype=torch.float64)
    packing = torch.full((3,), 0.7, dtype=torch.float64)
    kinds = torch.tensor([NP.KIND_ANCHOR, NP.KIND_ANCHOR, NP.KIND_CHILD])
    splits = torch.tensor([0, 1, 0])
    masks, floor = NP.select_rows(e, packing, kinds, splits, torch.zeros(3, dtype=torch.long), 1.0, _settings())
    assert floor[0] == 10.0
    assert masks['train'].tolist() == [False, False, True]


def test_no_energy_window_and_anchor_training():
    e = torch.tensor([0.0, 1e6], dtype=torch.float64)
    packing = torch.full((2,), 0.7, dtype=torch.float64)
    kinds = torch.tensor([NP.KIND_ANCHOR, NP.KIND_CHILD])
    splits = torch.zeros(2, dtype=torch.long)
    masks, _ = NP.select_rows(e, packing, kinds, splits, torch.zeros(2, dtype=torch.long), 1.0,
                              _settings(window_kT=None, train_kinds=['anchor', 'child']))
    assert masks['train'].tolist() == [True, True]


def test_settings_defaults_and_refusals():
    s = NP.settings(types.SimpleNamespace())
    assert s['window_kT'] == 6.0 and s['train_kinds'] == ('child',) and s['check_rows'] == NP.CHECK_ROWS
    from energies.molecular_crystal import PACKING_COEFF_WINDOW
    assert s['packing_window'] == tuple(PACKING_COEFF_WINDOW)
    with pytest.raises(ValueError, match='unknown keys'):
        _settings(window_kt=6)
    with pytest.raises(ValueError, match='train_kinds'):
        _settings(train_kinds=['noise'])


def test_check_rows_cover_every_stratum():
    kinds = [NP.KIND_ANCHOR] * 5 + [NP.KIND_CHILD] * 995
    splits = [0] * 3 + [1] * 2 + [0] * 900 + [1] * 95
    b = _batch(['a'] * 1000, [2] * 1000, [1] * 1000, noised_kind=kinds, noised_split=splits)
    idx = NP.check_rows(b, n=64, seed=0)
    k, s = torch.tensor(kinds)[idx], torch.tensor(splits)[idx]
    for kind in (NP.KIND_ANCHOR, NP.KIND_CHILD):
        for split in (0, 1):
            assert bool(((k == kind) & (s == split)).any()), (kind, split)
    assert len(set(idx.tolist())) == len(idx) <= 64


def test_identity_mismatch_names_each_difference():
    a = dict(energy_function='elj', lj_coeff=0.36, cutoff=10)
    b = dict(energy_function='elj', lj_coeff=1.0, cutoff=10)
    assert NP.identity_mismatch(a, a) == []
    assert NP.identity_mismatch(a, b) == ["lj_coeff: file 0.36, run 1.0"]


def test_compare_terms_passes_float_noise_and_fails_a_real_miss():
    stored = types.SimpleNamespace(elj=torch.tensor([-100.0, -50.0]), reduction_en=torch.zeros(2),
                                   packing_coeff=torch.tensor([0.7, 0.7]), cell_volume=torch.tensor([900.0, 800.0]))
    near = types.SimpleNamespace(elj=stored.elj + 0.05, reduction_en=stored.reduction_en,
                                 packing_coeff=stored.packing_coeff, cell_volume=stored.cell_volume + 0.01)
    ratio, _, _ = NP.compare_terms(stored, near, 'elj')
    assert ratio <= 1.0
    far = types.SimpleNamespace(**{**vars(near), 'elj': stored.elj + torch.tensor([0.0, 2.0])})
    ratio, worst, per = NP.compare_terms(stored, far, 'elj')
    assert ratio > 1.0 and abs(worst - 2.0) < 1e-6 and abs(per['elj'] - 2.0) < 1e-6


def test_identity_compares_the_current_definition_only():
    current = dict(energy_function='elj', lj_coeff=1.0)
    assert NP.identity_mismatch({**current, 'crystal_coordinates': 'latent'}, current) == []
    assert NP.identity_mismatch({'energy_function': 'elj'}, current) == ["lj_coeff: file None, run 1.0"]


def _record_modeller(T=2.5, **block):
    args = types.SimpleNamespace(energy_config=types.SimpleNamespace(temperature=T), prior_path='p.pt',
                                 noised_prior=types.SimpleNamespace(**block))
    return types.SimpleNamespace(args=args, NOISED_PRIOR_RECORD=_modeller_cls().NOISED_PRIOR_RECORD)


def _modeller_cls():
    from train import Modeller
    return Modeller


def test_a_block_on_a_plain_prior_is_inert_and_printed(capsys):
    rec = _modeller_cls()._noised_prior_record(_record_modeller(window_kT=6.0), None)
    assert rec == {'data/prior_prenoised': 0}
    assert 'noised_prior block inert: prior_path is not a pre-noised build' in capsys.readouterr().out
    m = _record_modeller()
    del m.args.noised_prior
    _modeller_cls()._noised_prior_record(m, None)
    assert 'inert' not in capsys.readouterr().out


def test_a_typo_in_the_block_refuses_a_plain_prior_too():
    with pytest.raises(ValueError, match='unknown keys'):
        _modeller_cls()._noised_prior_record(_record_modeller(window_kt=6.0), None)


def test_a_build_records_its_manifest_and_warns_on_another_temperature(capsys):
    manifest = dict(k=10, log_lo=-2.5, log_hi=-1.5, window_kT_target=6.0, temperature=2.5)
    rec = _modeller_cls()._noised_prior_record(_record_modeller(T=2.5), manifest)
    assert rec == {'data/prior_prenoised': 1, 'data/prior_noised_k': 10, 'data/prior_noised_log_lo': -2.5,
                   'data/prior_noised_log_hi': -1.5, 'data/prior_noised_window_kT_target': 6.0,
                   'data/prior_noised_build_T': 2.5}
    assert 'WARNING' not in capsys.readouterr().out
    _modeller_cls()._noised_prior_record(_record_modeller(T=6.9), manifest)
    assert 'built at T 2.5' in capsys.readouterr().out


def test_harvest_keeps_high_error_rows_and_records_and_drops_the_rest():
    resid = torch.tensor([0.2, -3.0, 0.1, float('nan'), 0.0])
    log_r = torch.tensor([-10.0, -10.0, -1.0, -10.0, float('-inf')])
    floor = torch.tensor([0.0, 0.0, 3.0, 0.0, 0.0])          # row 2: energy 1 x T 2 = 2 <= 3, a record
    keep = NP.harvest_mask(resid, log_r, 2.0, 1.0, floor)
    assert keep.tolist() == [False, True, True, False, False]
    assert NP.harvest_mask(resid, log_r, 2.0, 1.0).tolist() == [False, True, False, False, False]
