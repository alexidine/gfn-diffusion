"""protocol_card: the HTML card of a config's training protocol (python -m config_snapshot <cfg> --card PATH). What
each test would catch:

  * a declared mk_dev protocol that fails to render, or a stage panel missing;
  * the card disagreeing with describe_stage about which parameters a branch trains (both read the same overlay);
  * a lambda anneal shown as live when lambda_mix already sits at its target;
  * vg_by_condition, a grouping switch, shown as a loss term instead of a note on the terms it groups;
  * a balance whose rules all boost the default mix and whose anneal is inert shown as live;
  * the stage-1 MLE draw and the stage-2 prior buffer given the same row source.

    python -m pytest -q tests/config/test_protocol_card.py
"""
from pathlib import Path

import pytest

import config_snapshot as cs
import protocol_card as pc

pytestmark = pytest.mark.fast
CANONICAL = Path(__file__).resolve().parents[2] / 'configs' / 'mk_dev.yaml'


@pytest.fixture(scope='module')
def args():
    snap, issues, a = cs._contract(str(CANONICAL))
    assert a is not None, snap.get('load_error')
    return a


def test_every_mk_dev_protocol_renders_one_panel_per_stage(args):
    for name, proto in cs._to_plain(args.protocols).items():
        html = pc.render_html(args, name)
        stages = proto['stages']
        assert html.count('class="stage"') == len(stages), name
        for s in stages:
            assert f'>{s["name"]}<' in html, (name, s['name'])


def test_trained_parameters_agree_with_the_prose(args):
    for name, proto in cs._to_plain(args.protocols).items():
        _, cards = pc.protocol_cards(args, name)
        prose = cs.describe_protocol(args, name).split('\n')
        for card, para in zip(cards, prose):
            for b in card['branches']:
                if b['inert']:
                    assert f"{b['mode']} carries no loss weight" in para, (name, b['mode'])
                    continue
                want = {('policy', 'log Z'): 'the policy and log Z', ('policy',): 'the policy only',
                        ('log Z',): 'log Z only', (): 'nothing'}[tuple(b['trains'])]
                assert f"{b['mode']} trains {want}" in para or f"{b['mode']} alone trains {want}" in para, \
                    (name, b['mode'], b['trains'], para)


def test_anneal_at_its_target_is_marked_inert(args):
    _, cards = pc.protocol_cards(args, 'conditional_vargrad')
    bal = [m for c in cards for m in c['mech'] if m[0] == 'balance']
    assert bal, 'conditional_vargrad declares a lambda anneal'
    if float(args.energy_config.lambda_mix) >= 1.0:
        assert all(inert for _, _, inert in bal) and 'inert' in bal[0][1]


def test_vg_by_condition_is_not_a_term_on_its_own():
    assert pc._terms({'mle': 1.0, 'vg_by_condition': 1.0}) == ['mle 1']
    assert pc._terms({'vg_lb': 1.0, 'vg_by_condition': 1.0}) == ['vg_lb 1 · by condition']
    assert pc._terms({'emp_z': 1.0, 'vg_by_condition': 1.0}, 'fwd') == ['emp_z 1 · by condition']
    assert pc._terms({'emp_z': 1.0, 'vg_by_condition': 1.0}, 'bwd') == ['emp_z 1']


def test_a_fixed_split_balance_with_an_inert_anneal_reads_off(args):
    _, cards = pc.protocol_cards(args, 'conditional_vargrad')
    bal = [m for c in cards for m in c['mech'] if m[0] == 'balance']
    if float(args.energy_config.lambda_mix) >= 1.0:
        assert bal[0][1].startswith('OFF'), bal[0][1]


def test_each_branch_says_where_its_rows_come_from(args):
    _, cards = pc.protocol_cards(args, 'conditional_vargrad')
    rows = {b['mode']: b['rows'] for c in cards for b in c['branches']}
    assert rows['replay'] == 'replay buffer' and rows['fwd'] == 'own rollouts'
    chips = [b['rows'] for c in cards for b in c['branches'] if b['mode'] == 'bwd']
    assert chips == ['prior dataset (MLE draw)', 'prior buffer']
