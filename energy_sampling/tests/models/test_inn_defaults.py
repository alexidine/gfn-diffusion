"""
CPU tests for how inn_modeller.py fills the options left unset (inn_modeller.resolve_auto_options).

WHAT THIS FILE IS FOR. Four options default per route: on an unconditional run bounded linear rows, tied knot
derivatives, a 1e-7 bin-height floor and replay at 0.1; on a conditional run, which refuses all four, off. With
--init_from the density options come from the checkpoint instead, so an older checkpoint is rebuilt as the density it
was trained as: a key the checkpoint lacks means the setting it ran under before the key existed. Every failure here is
silent in a run (a checkpoint loaded into the wrong density still produces numbers), so each case is checked explicitly,
and an explicit flag must survive resolution untouched.

    python test_inn_defaults.py
"""
import os
import sys
import types

_here = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))   # tests/<area>/x.py -> energy_sampling/
for p in (_here, os.path.dirname(_here)):
    p = os.path.abspath(p)
    if p not in sys.path:
        sys.path.insert(0, p)

import inn_modeller as im  # noqa: E402


def _unset():
    return types.SimpleNamespace(bounded_rows='auto', min_bin_height='auto', tied_derivatives=None, tied_factor=None,
                                 bank_frac=None)


def _settings(inn):
    return inn.bounded_rows, inn.min_bin_height, inn.tied_derivatives, inn.tied_factor, inn.bank_frac


def test_parser_leaves_the_four_options_unset():
    a = im.build_parser().parse_args(['--config', 'x', '--run_name', 'y'])
    assert (a.bounded_rows, a.min_bin_height, a.tied_derivatives, a.tied_factor, a.bank_frac) == \
        ('auto', 'auto', None, None, None)
    b = im.build_parser().parse_args(['--config', 'x', '--run_name', 'y', '--no-tied_derivatives', '--no-bank_additive'])
    assert b.tied_derivatives is False and b.bank_additive is False


def test_unconditional_defaults():
    inn = _unset()
    im.resolve_auto_options(inn, conditional=False)
    assert _settings(inn) == ('linear', 1e-7, True, 2.0, 0.1)


def test_conditional_route_is_off():
    inn = _unset()
    im.resolve_auto_options(inn, conditional=True)
    assert _settings(inn) == (None, None, False, 2.0, 0.0)


def test_old_checkpoint_keeps_its_density():
    # a checkpoint saved before the keys existed: unbounded, 1e-3 everywhere, free derivatives; replay is the route's
    inn = _unset()
    im.resolve_auto_options(inn, conditional=False, ck_inn={'n_bins': 32})
    assert _settings(inn) == (None, None, False, 2.0, 0.1)


def test_checkpoint_settings_are_copied():
    inn = _unset()
    im.resolve_auto_options(inn, conditional=False, ck_inn=dict(bounded_rows='0,1,2', min_bin_height=1e-5,
                                                                tied_derivatives=True, tied_factor=1.0))
    assert _settings(inn) == ('0,1,2', 1e-5, True, 1.0, 0.1)


def test_explicit_values_survive():
    inn = types.SimpleNamespace(bounded_rows='none', min_bin_height='1e-5', tied_derivatives=False, tied_factor=None,
                                bank_frac=0.0)
    account = im.resolve_auto_options(inn, conditional=False, ck_inn={'tied_derivatives': True})
    assert _settings(inn) == (None, 1e-5, False, 2.0, 0.0)
    assert account == ''   # nothing was left unset, so nothing is reported as resolved


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok', name)
