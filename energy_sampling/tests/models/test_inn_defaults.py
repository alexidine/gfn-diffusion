"""
CPU tests for inn_modeller.py's defaults and how it fills the options left unset (inn_modeller.resolve_auto_options).

WHAT THIS FILE IS FOR. The architecture defaults are one transformer block of 32 bins. Five options default per route:
on an unconditional run bounded linear rows, tied knot derivatives, a 1e-7 bin-height floor, replay at 0.1 and, on a
one-block flow over the 12-row latent, the row order Z1_ORDER; on a conditional run, which refuses them, off. With
--init_from the density options and the order come from the checkpoint instead, so an older checkpoint is rebuilt as
the density it was trained as: a key the checkpoint lacks means the setting it ran under before the key existed. Every
failure here is silent in a run (a checkpoint loaded into the wrong density still produces numbers), so each case is
checked explicitly, and an explicit flag must survive resolution untouched.

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

LIVE12 = list(range(12))


def _unset(n_blocks=1):
    return types.SimpleNamespace(bounded_rows='auto', min_bin_height='auto', tied_derivatives=None, tied_factor=None,
                                 bank_frac=None, block_orders='auto', n_blocks=n_blocks)


def _settings(inn):
    return inn.bounded_rows, inn.min_bin_height, inn.tied_derivatives, inn.tied_factor, inn.bank_frac


def test_parser_defaults():
    a = im.build_parser().parse_args(['--config', 'x', '--run_name', 'y'])
    assert (a.bounded_rows, a.min_bin_height, a.tied_derivatives, a.tied_factor, a.bank_frac, a.block_orders) == \
        ('auto', 'auto', None, None, None, 'auto')
    assert (a.n_blocks, a.n_bins, a.conditioner) == (1, 32, 'transformer')
    b = im.build_parser().parse_args(['--config', 'x', '--run_name', 'y', '--no-tied_derivatives', '--no-bank_additive'])
    assert b.tied_derivatives is False and b.bank_additive is False


def test_unconditional_defaults():
    inn = _unset()
    im.resolve_auto_options(inn, conditional=False, live_rows=LIVE12)
    assert _settings(inn) == ('linear', 1e-7, True, 2.0, 0.1)
    assert inn.block_orders == im.Z1_ORDER


def test_conditional_route_is_off():
    inn = _unset()
    im.resolve_auto_options(inn, conditional=True, live_rows=LIVE12)
    assert _settings(inn) == (None, None, False, 2.0, 0.0)
    assert inn.block_orders is None


def test_z1_order_only_on_one_block_over_the_12_row_latent():
    for live, n_blocks in ((LIVE12[:-1], 1), (LIVE12 + [12, 13], 1), (LIVE12, 2)):
        inn = _unset(n_blocks)
        im.resolve_auto_options(inn, conditional=False, live_rows=live)
        assert inn.block_orders is None, (live, n_blocks)


def test_old_checkpoint_keeps_its_density():
    # a checkpoint saved before the keys existed: unbounded, 1e-3 everywhere, free derivatives, alternating orders;
    # replay is the route's
    inn = _unset()
    im.resolve_auto_options(inn, conditional=False, ck_inn={'n_bins': 32}, live_rows=LIVE12)
    assert _settings(inn) == (None, None, False, 2.0, 0.1)
    assert inn.block_orders is None


def test_checkpoint_settings_are_copied():
    inn = _unset()
    im.resolve_auto_options(inn, conditional=False, ck_inn=dict(bounded_rows='0,1,2', min_bin_height=1e-5,
                                                                tied_derivatives=True, tied_factor=1.0,
                                                                block_orders='rev'), live_rows=LIVE12)
    assert _settings(inn) == ('0,1,2', 1e-5, True, 1.0, 0.1)
    assert inn.block_orders == 'rev'


def test_explicit_values_survive():
    inn = types.SimpleNamespace(bounded_rows='none', min_bin_height='1e-5', tied_derivatives=False, tied_factor=None,
                                bank_frac=0.0, block_orders='id', n_blocks=1)
    account = im.resolve_auto_options(inn, conditional=False, ck_inn={'tied_derivatives': True}, live_rows=LIVE12)
    assert _settings(inn) == (None, 1e-5, False, 2.0, 0.0)
    assert inn.block_orders == 'id'
    assert account == ''   # nothing was left unset, so nothing is reported as resolved
    inn = _unset()
    inn.block_orders = 'none'
    im.resolve_auto_options(inn, conditional=False, live_rows=LIVE12)
    assert inn.block_orders is None


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok', name)
