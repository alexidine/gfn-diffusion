"""
Numeric protocol fields are refused AT PARSE when they are not numbers.

WHAT WAS WRONG. `Stage._parse_exit` checked only key names, so an exit term's
`above`/`below` reached `_term_passes` as whatever YAML produced and failed on
the first tick that resolved the metric -- mid-run -- with

    TypeError: '<' not supported between instances of 'float' and 'str'

naming neither the stage nor the metric. Trigger seen 2026-09-11: `below:
1.0e6`. PyYAML resolves YAML 1.1 floats, which need a dot AND a signed exponent,
so `1.0e6` loads as the STRING '1.0e6'; only `1.0e+6` is a float.

Balance rules and hot_lr_sensor had the neighbouring gap: they pass through
float()/int() (at tick time for rules), which fails late on garbage and quietly
coerces `True` and numeric strings.

`pytest tests/protocol/test_numeric_fields_fail_at_parse.py -q`
"""

import pytest
import yaml

from protocol import Stage

pytestmark = pytest.mark.fast


def stage(**extra):
    return Stage({'name': 's0', 'train_mode': 'bwd', 'bwd_sampling_mode': 'dataset',
                  **extra}, 0)


def exit_stage(term):
    return stage(exit=[term])


# ---------------------------------------------------------------------------
# The trigger itself
# ---------------------------------------------------------------------------

def test_pyyaml_reads_unsigned_exponent_as_a_string():
    """The premise: if PyYAML ever starts resolving `1.0e6`, the hint is stale."""
    assert yaml.safe_load('x: 1.0e6')['x'] == '1.0e6'
    assert yaml.safe_load('x: 1.0e+6')['x'] == 1.0e6


def test_the_observed_yaml_fails_at_parse_naming_stage_metric_and_value():
    term = yaml.safe_load('{metric: bwd/tbc, below: 1.0e6, patience: 3}')
    with pytest.raises(ValueError) as e:
        exit_stage(term)
    msg = str(e.value)
    assert "stage 's0'" in msg
    assert "'bwd/tbc'" in msg
    assert "'1.0e6'" in msg
    assert '1.0e+6' in msg  # the fix, spelled out


def test_the_signed_exponent_form_loads():
    term = yaml.safe_load('{metric: bwd/tbc, below: 1.0e+6, patience: 3}')
    s = exit_stage(term)
    assert s.exit == [{'metric': 'bwd/tbc', 'below': 1.0e6, 'patience': 3}]


# ---------------------------------------------------------------------------
# Exit terms
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('side', ['above', 'below'])
@pytest.mark.parametrize('bad', ['0.5', 'abc', True, False, None, [1.0], float('nan')])
def test_exit_bar_must_be_a_real_number(side, bad):
    with pytest.raises(ValueError, match=f"'{side}' must be a number"):
        exit_stage({'metric': 'bwd/tbc', side: bad})


@pytest.mark.parametrize('good', [0, 2, 0.015, -3.5, float('inf')])
def test_exit_bar_accepts_ints_and_floats(good):
    assert exit_stage({'metric': 'bwd/tbc', 'above': good}).exit[0]['above'] == good


@pytest.mark.parametrize('bad', [-1, 2.5, 5.0, '5', True])
def test_exit_patience_must_be_a_non_negative_int(bad):
    with pytest.raises(ValueError, match="'patience' must be an integer >= 0"):
        exit_stage({'metric': 'bwd/tbc', 'below': 2.0, 'patience': bad})


@pytest.mark.parametrize('good', [0, 1, 5])
def test_exit_patience_accepts_non_negative_ints(good):
    assert exit_stage({'metric': 'bwd/tbc', 'below': 2.0, 'patience': good}).exit[0]['patience'] == good


@pytest.mark.parametrize('bad', [None, '', 3])
def test_exit_metric_must_be_a_string(bad):
    term = {'below': 2.0} if bad is None else {'metric': bad, 'below': 2.0}
    with pytest.raises(ValueError, match="'metric' must be a non-empty string"):
        exit_stage(term)


# ---------------------------------------------------------------------------
# Balance rules (lexicographic) and anneal_coeffs
# ---------------------------------------------------------------------------

def balance_stage(rule=None, **node):
    rules = [{'boost': 'fwd', **rule}] if rule else []
    bal = {'kind': 'lexicographic', 'default_boost': 'replay', 'rules': rules, **node}
    return stage(balance=bal)


@pytest.mark.parametrize('key,rule', [
    ('above', {'metric': 'fwd/tb_err', 'above': '1.0e6'}),
    ('below', {'metric': 'gates/r2', 'below': True}),
    ('margin', {'metric': 'fwd/tb_err', 'relative': 'best', 'margin': '1.3'}),
    ('drift', {'metric': 'fwd/tb_err', 'relative': 'best', 'drift': 'x'}),
    ('floor', {'metric': 'fwd/tb_err', 'relative': 'best', 'floor': None}),
    ('anneal.rate', {'metric': 'fwd/tb_err', 'above': 1.0, 'anneal': {'rate': '0.9'}}),
    ('anneal.min', {'metric': 'fwd/tb_err', 'above': 1.0, 'anneal': {'min': True}}),
])
def test_balance_rule_numeric_fields_must_be_numbers(key, rule):
    with pytest.raises(ValueError, match=f"'{key}' must be a number") as e:
        balance_stage(rule)
    assert "stage 's0' rule 0" in str(e.value)


def test_balance_rule_anneal_min_may_name_a_metric():
    """`min` is a number OR a live-floor metric name; the name must survive."""
    s = balance_stage({'metric': 'replay/tb_err', 'above': 1.0,
                       'anneal': {'rate': 0.9, 'min': 'fwd/scatter_err'}})
    assert s.balance['rules'][0]['anneal']['min'] == 'fwd/scatter_err'


def test_anneal_coeffs_target_must_be_a_number():
    with pytest.raises(ValueError, match=r"anneal_coeffs\.bounding_coeff\.target must be a number"):
        balance_stage(anneal_coeffs={'bounding_coeff': {'target': '1.0e3'}})


# ---------------------------------------------------------------------------
# hot_lr_sensor
# ---------------------------------------------------------------------------

HOT = {'channel': 'bwd/mle', 'form': 'absolute', 'rows': 31, 'above': 5.0}


@pytest.mark.parametrize('key,bad,match', [
    ('above', '5.0', r'hot_lr_sensor\.above must be a number'),
    ('above', True, r'hot_lr_sensor\.above must be a number'),
    ('floor_percentile', '10', r'hot_lr_sensor\.floor_percentile must be a number'),
    ('rows', 31.0, r'hot_lr_sensor\.rows must be an integer'),
    ('rows', '31', r'hot_lr_sensor\.rows must be an integer'),
    ('row_steps', True, r'hot_lr_sensor\.row_steps must be an integer'),
])
def test_hot_lr_sensor_numeric_fields_must_be_numbers(key, bad, match):
    with pytest.raises(ValueError, match=match):
        stage(hot_lr_sensor={**HOT, key: bad})


def test_hot_lr_sensor_valid_block_still_parses():
    assert stage(hot_lr_sensor=HOT).hot_lr_sensor['above'] == 5.0
