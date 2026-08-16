"""
Mode safety: keys that are documented INERT in the canonical config must be inert.

WHAT THIS IS FOR. Phase 1 of the infrastructure stabilization pass consolidates
the canonical config so conditional and unconditional settings coexist and only
the selected mode activates. The requirement and the invariant are the same
sentence -- *inactive config modes must not influence execution* -- so it is
written down here as a test BEFORE the consolidation, not after. A mode-safety
test written afterwards is built to pass.

EACH CASE BELOW IS A COMMENT IN configs/mk_dev.yaml MADE EXECUTABLE. The config
says things like "inert while vector_conditioning is false" and
"applies only in stages declaring the flag". Those claims are load-bearing and
were checked by hand once; here they are checked every run of the suite.

WHAT IT PROVES, AND WHAT IT DOES NOT. This is a STATIC check: it perturbs a key
and asserts nothing DERIVED moves -- no resolved value, no parsed stage, no
effective loss coefficient. That catches leakage through the derivation layer,
which is where a config-shaped mistake usually lives.

It does NOT prove the trainer never reads the key at runtime. Proving that needs
a run, and the run-level check belongs with the smoke harness that tier C of the
acceptance criterion also needs. Until that exists, this file is evidence, not
proof, and the docstring says so rather than letting a green suite imply more
than it earned.

Run: python -m pytest test_mode_safety.py -q
"""

import copy
from pathlib import Path

import pytest
import yaml

import config_snapshot as cs

HERE = Path(__file__).parent
CANONICAL = HERE / 'configs' / 'mk_dev.yaml'


@pytest.fixture(scope='module')
def raw():
    return yaml.safe_load(CANONICAL.read_text(encoding='utf-8'))


@pytest.fixture(scope='module')
def base_snap():
    return cs.snapshot(str(CANONICAL))


def _set(cfg: dict, dotted: str, value):
    node = cfg
    parts = dotted.split('.')
    for p in parts[:-1]:
        node = node[p]
    assert parts[-1] in node, f'{dotted} is not in the canonical config'
    node[parts[-1]] = value


def derived_impact(base_snap, raw, dotted, value, tmp_path, name):
    """Paths whose value moved when `dotted` was set to `value`, EXCLUDING the
    key itself.

    An inert key perturbs exactly one path: its own. Anything else in the list is
    the key reaching something -- a resolved value, a parsed stage, an effective
    coefficient -- while the mode that owns it is switched off."""
    new = copy.deepcopy(raw)
    _set(new, dotted, value)
    p = tmp_path / name
    p.write_text(yaml.safe_dump(new, sort_keys=False), encoding='utf-8')
    cmp = cs.compare(base_snap, cs.snapshot(str(p)))
    own = f'config.{dotted}'
    return [(path, o, n) for path, o, n in cmp.changed if path != own]


# The canonical config is UNCONDITIONAL MOLECULE (elj, vector_conditioning and
# molecule_conditioning both false, z_calibration.mode rollout). Each entry is a
# key the config documents as inert under exactly those conditions, paired with a
# value that is a real change to it.
INERT_UNDER_UNCONDITIONAL = [
    # "inert while vector_conditioning is false"
    ('vector_conditioning_dim', 7),
    # "held-out conditional eval batch; inert while test_molecules_path is null"
    ('test_eval_num_samples', 4321),
    # "fit-error-weighted forward mol draws; applies only in stages declaring the flag"
    ('condition_log_z.weighted_condition_sampling_temperature', 0.9),
    ('condition_log_z.weighted_condition_sampling_uniform_beta', 0.1),
    ('condition_log_z.weighted_condition_sampling_clip_quantile', 0.5),
    # "sensor: worst only -- inert on the other three sensors" (this route: pooled)
    ('z_calibration.sensor_quantile', 0.9),
    # "mode: regression only (inert on this route, which runs rollout)"
    ('z_calibration.freshness_half_life_steps', 999.0),
    ('z_calibration.se2_floor', 0.99),
    ('z_calibration.holdout_modulus', 3),
    # "uniform fraction for stages with weighted_bwd_sampling; no stage here
    #  declares that flag"
    ('buffers.prior_buffer.weighted_bwd_beta', 0.1),
    # "degrades to pooled on unconditional runs"
    ('conditional_worst_quantile', 0.9),
]


@pytest.mark.parametrize('dotted,value', INERT_UNDER_UNCONDITIONAL,
                         ids=[d for d, _ in INERT_UNDER_UNCONDITIONAL])
def test_documented_inert_key_has_no_derived_effect(dotted, value, raw, base_snap,
                                                    tmp_path):
    impact = derived_impact(base_snap, raw, dotted, value, tmp_path,
                            f'{dotted.replace(".", "_")}.yaml')
    assert impact == [], (
        f'{dotted} is documented inert on this route but moved: '
        + '; '.join(f'{p}: {o!r}->{n!r}' for p, o, n in impact))


# ---------------------------------------------------------------------------
# The load gate: `auto` must yield to an adaptive scheme, end to end
# ---------------------------------------------------------------------------

def _load(raw_cfg):
    """Full load path: preflight + derived resolution, output suppressed."""
    import contextlib
    import io

    import utils
    with contextlib.redirect_stdout(io.StringIO()):
        return utils.resolve_derived_config(
            utils.preflight_config(utils.dict2namespace(raw_cfg)))


def test_canonical_config_loads(raw):
    assert _load(copy.deepcopy(raw)) is not None


def test_auto_lr_without_a_sensor_is_refused_at_load(raw):
    """A config that reads as adaptive and trains at a fixed seed must not start.
    This is the gate, not just the report -- the whole cost of the failure is that
    it is invisible for the length of a run."""
    cfg = copy.deepcopy(raw)
    for st in cfg['protocol']['stages']:
        st.pop('lr_sensor', None)
    with pytest.raises(ValueError, match='adaptive sensor'):
        _load(cfg)


def test_explicit_float_lrs_need_no_sensor_at_load(raw):
    """A float is a fixed peak by intent: it takes the warmup envelope and
    divergence handling but never peak_scale, so it has nothing to yield to."""
    cfg = copy.deepcopy(raw)
    for st in cfg['protocol']['stages']:
        st.pop('lr_sensor', None)
    for k in ('lr_policy', 'lr_back', 'lr_replay', 'lr_fused'):
        cfg[k] = 3.0e-4
    assert _load(cfg) is not None


def test_hyper_only_config_loads_with_ray_calibration_disabled(raw):
    """`hyper` does not use ray_calibration at all. The previous gate tested
    `ray_calibration.enabled` and would have rejected this legitimate config."""
    cfg = copy.deepcopy(raw)
    for st in cfg['protocol']['stages']:
        st['lr_sensor'] = {'kind': 'hyper', 'beta': 0.05}
    cfg['ray_calibration']['enabled'] = False
    assert _load(cfg) is not None


# ---------------------------------------------------------------------------
# The mutation: the method must be able to SEE a key that is not inert
# ---------------------------------------------------------------------------

def test_the_check_detects_a_key_that_is_genuinely_live(raw, base_snap, tmp_path):
    """Without this, every assertion above would pass on a comparator that
    reported nothing. `integrator.T` feeds the derived gradient_norm_clip, so
    perturbing it must show a derived consequence."""
    new = copy.deepcopy(raw)
    new['integrator']['T'] = 25
    new['eval_T'] = 25                      # preflight requires agreement
    p = tmp_path / 'live.yaml'
    p.write_text(yaml.safe_dump(new, sort_keys=False), encoding='utf-8')
    changed = cs.compare(base_snap, cs.snapshot(str(p))).changed
    paths = [x[0] for x in changed]
    assert 'config.gradient_norm_clip' in paths, paths


def test_a_live_loss_coefficient_shows_a_derived_effect(raw, base_snap, tmp_path):
    """Second mutation, on the effective-coefficient path specifically: a base
    coefficient reaches every stage that does not override it."""
    impact = derived_impact(base_snap, raw, 'bwd_loss_coeffs.traj_grads', 0.0,
                            tmp_path, 'livecoeff.yaml')
    assert any('effective_loss_coeffs' in p for p, _, _ in impact), impact
