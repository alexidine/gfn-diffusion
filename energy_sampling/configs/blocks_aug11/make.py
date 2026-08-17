"""
blocks_aug11 -- condition_block_m vs repeats for condition-grouped VarGrad.

    python configs/blocks_aug11/make.py             # write the arm configs + INDEX
    python configs/blocks_aug11/make.py --preflight # verify data + checkpoints exist

=============================================================================
THE QUESTION
=============================================================================
Condition-grouped VarGrad needs >= min_group_count (2) rows per condition or the
row contributes nothing: vg_loss is identically zero on singleton groups and
emp_z masks them out. With ~900 conditions against a 500-row draw most conditions
land 0-1 times, so group size has to be manufactured. There are two ways, and
they are NOT the same measurement:

    repeats K           K rollouts from the SAME terminal. Within-terminal.
                        gflownet_losses calls this "TBC in disguise (reward
                        cancels)" for same-terminal tiles. Costs K x rows.
    condition_block_m   m DISTINCT terminals of one condition. Cross-terminal,
                        which is the axis condition-grouped VarGrad actually
                        estimates. Costs NOTHING: _sample_condition_blocked_indices
                        returns exactly batch_size rows.

Prediction: blocks dominate repeats at equal group size, because the group's
within-spread then samples the terminal axis rather than only trajectory noise.
Arms A/B test that at matched group size; A/C tests whether stacking both helps
at matched cost.

=============================================================================
ARMS  (fwd repeats is FIXED at 2 -- fwd has no block lever, so only bwd varies)
=============================================================================
    arm   block_m  bwd K   bwd rows   groups   size   members
    a         1       2      2B        ~B       2     same terminal   (= ca11)
    b         2       1       B         B/2     2     distinct terminals
    c         2       2      2B         B/2     4     2 terminals x 2 rollouts

    a vs b  matched group size, b at HALF the rows -- the core contrast
    a vs c  matched rows, c has bigger and better-composed groups

Arm d (block_m 4, K 1) is the depth probe and is deliberately NOT in this pass;
add it to ARMS if a/b separates.

=============================================================================
READING IT
=============================================================================
vg_live_frac (added to gflownet_losses.condition_group_stats for this battery)
is the mechanism check: it is the fraction of rows carrying any VarGrad
gradient. Confirm it is ~1.0 on every arm BEFORE comparing outcomes -- if an arm
is running at live_frac 0.3 the comparison is about occupancy, not about the
axis. Then read bwd/logw_std_within (the component the estimator targets) and
bwd/vg_lb, against train_step_time for the cost side.

Seed is shared and single. aug02 measured ~35% seed spread on this codebase, so
treat sub-35% gaps as direction only and reseed whatever looks decisive.
"""

import argparse
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))

from configs.generate_configs import overwrite_nested_dict  # noqa: E402

STAGE = 'var_conditioning'

# (name, condition_block_m, bwd repeats, fwd repeats)
#
# `repeats` MEANS OPPOSITE THINGS on the two branches, which is why fwd repeats
# is a depth knob here and bwd repeats is not:
#   bwd repeats tiles a BUFFER ROW  -> same terminal, K trajectories. log R is
#       constant in the group so it CANCELS: this is TBC (get_tbc_loss is
#       explicitly "the data-driven, reward-free VarGrad", arXiv:2209.02606
#       eq. 33). Adds no cross-terminal signal.
#   fwd repeats tiles a CONDITION (condition_samples broadcasts one sampled
#       condition across the tile) -> K DISTINCT terminals, since each tile runs
#       its own forward trajectory. log R varies, so it SURVIVES the centring.
#       This is the fwd-side analogue of condition_block_m.
# Cost asymmetry: bwd blocks are free (still batch_size rows); fwd depth pays a
# full rollout AND an energy evaluation per extra terminal.
ARMS = [
    ('blk_a_repeats', 1, 2.0, 2.0),
    ('blk_b_blocks', 2, 1.0, 2.0),
    ('blk_c_both', 2, 2.0, 2.0),
    # D: bwd cross-terminal DEPTH. Free -- still 500 bwd rows. At m=2 a group's
    # two centred deviations are perfectly anti-correlated, so each group carries
    # exactly one independent datum (0.5 dof/row); m=4 gives 0.75. Expected to
    # turn over when condition REVISIT rate starves condition_log_z (min_visits
    # 20, half_life_visits 28): 500/m conditions per batch out of ~900, so every
    # ~3.6 batches at m=2 but ~7 at m=4.
    ('blk_d_deep_blocks', 4, 1.0, 2.0),
    # E: the same depth move on the FWD branch, which is the expensive one.
    # Isomorphic to raising block_m, so D vs E tests whether the branch matters
    # or only the group depth.
    ('blk_e_fwd_depth', 2, 1.0, 3.0),
]


def _stage(cfg, name=STAGE):
    """By name, never by index -- an inserted stage would silently retarget."""
    for st in cfg['protocol']['stages']:
        if st['name'] == name:
            return st
    raise KeyError(f'stage {name!r} not in protocol')


def build_config(base, name, block_m, bwd_repeats, fwd_repeats):
    cfg = overwrite_nested_dict(yaml.safe_load(yaml.safe_dump(base)), {'run_name': name})
    cfg['bwd_loss_coeffs']['condition_block_m'] = block_m
    _stage(cfg)['loss_coeffs']['bwd']['repeats'] = bwd_repeats
    _stage(cfg)['loss_coeffs']['fwd']['repeats'] = fwd_repeats
    return cfg


def assert_pinned_resume(cfg, name):
    """Every arm warm-starts from the SAME ca11 phase-1 exit, weights only.

    Asserted rather than trusted: a null checkpoint_name silently retrains phase 1
    per arm, and load_weights_only false would carry ca11's optimizer state and
    step index into the run. Both are invisible in the results.
    """
    assert cfg['continue_from_checkpoint'] is False, f'{name}: continue_from_checkpoint must be false'
    assert cfg['checkpoint_name'] and 'phase1_exit' in cfg['checkpoint_name'], \
        f'{name}: checkpoint_name must pin the shared phase1_exit'
    assert cfg['prior_model_name'], f'{name}: prior_model_name must be set or train_prior will not skip'
    assert cfg['load_weights_only'] is True, f'{name}: load_weights_only must be true'


def assert_controlled(cfg, name):
    """Everything the contrast is NOT about has to be identical and correct."""
    st = _stage(cfg)
    assert st['flags']['weighted_bwd_sampling'] is False, \
        f'{name}: weighted_bwd_sampling must be off -- blocked draws bypass it, confounding the arms'
    # fwd repeats is a cross-terminal DEPTH knob (see ARMS), so it is a variable
    # rather than a control -- but only arm E is allowed to move it, or the fwd
    # and bwd depth effects would be confounded within one arm.
    fwd_k = st['loss_coeffs']['fwd']['repeats']
    assert fwd_k == (3.0 if name.endswith('fwd_depth') else 2.0), \
        f'{name}: fwd repeats {fwd_k} -- only the fwd-depth arm may leave the 2.0 reference'
    assert cfg['bwd_loss_coeffs']['vg_by_condition'] > 0, \
        f'{name}: bwd vg_by_condition must be on or the condition-grouped path never runs'
    assert st['loss_coeffs']['bwd']['vg_lb'] > 0, \
        f'{name}: bwd vg_lb must be > 0 -- it is the gate that lets condition_block_m through at all'
    assert cfg['grow_batch_size'] is False and cfg['batch_size'] == cfg['max_batch_size'], \
        f'{name}: batch must be pinned -- collision rate ~B^2/2N is the free cross-terminal baseline'


def assert_distinct(configs):
    """No arm may be a duplicate written by omission."""
    keys = {}
    for name, cfg in configs:
        k = (cfg['bwd_loss_coeffs']['condition_block_m'],
             _stage(cfg)['loss_coeffs']['bwd']['repeats'],
             _stage(cfg)['loss_coeffs']['fwd']['repeats'])
        assert k not in keys, f'{name} duplicates {keys[k]}: both are m/Kbwd/Kfwd = {k}'
        keys[k] = name


def preflight(configs):
    missing = []
    for name, cfg in configs:
        for key in ('prior_path', 'molecules_path', 'test_molecules_path'):
            path = cfg.get(key)
            if path and not Path(path).exists():
                missing.append(f'  {name}.{key}: {path}')
        ckpt_dir = Path(cfg['checkpoints_dir'])
        for key in ('checkpoint_name', 'prior_model_name'):
            if not (ckpt_dir / cfg[key]).exists():
                missing.append(f'  {name}.{key}: {ckpt_dir / cfg[key]}')
    if missing:
        print('MISSING:\n' + '\n'.join(sorted(set(missing))))
        return 1
    print(f'preflight OK: data + checkpoints for {len(configs)} arms all exist')
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preflight', action='store_true')
    args = ap.parse_args()

    base = yaml.safe_load((HERE / 'base.yaml').read_text())

    configs = []
    for name, block_m, bwd_repeats, fwd_repeats in ARMS:
        cfg = build_config(base, name, block_m, bwd_repeats, fwd_repeats)
        assert_pinned_resume(cfg, name)
        assert_controlled(cfg, name)
        configs.append((name, cfg))
    assert_distinct(configs)

    if args.preflight:
        raise SystemExit(preflight(configs))

    rows = ['name\tblock_m\tbwd_K\tfwd_K\tbwd_rows\tfwd_rows\tbwd_group\tfwd_group_approx']
    for name, cfg in configs:
        out = HERE / f'{name}.yaml'
        out.write_text(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False))
        m = cfg['bwd_loss_coeffs']['condition_block_m']
        kb = _stage(cfg)['loss_coeffs']['bwd']['repeats']
        kf = _stage(cfg)['loss_coeffs']['fwd']['repeats']
        b = cfg['batch_size']
        # fwd group size is only APPROXIMATE: fwd conditions are sampled, so the
        # realised size is kf x (1 + birthday collisions), which is why arm A
        # measured 3.13 rather than its nominal 2 on the bwd side.
        rows.append(f"{name}\t{m}\t{kb:g}\t{kf:g}\t{b * kb:g}\t{b * kf:g}\t"
                    f"{max(m, 1) * kb:g}\t~{kf:g}+")
        print(f'wrote {out}')
    (HERE / 'INDEX.tsv').write_text('\n'.join(rows) + '\n')
    print(f'wrote {HERE / "INDEX.tsv"}')


if __name__ == '__main__':
    main()
