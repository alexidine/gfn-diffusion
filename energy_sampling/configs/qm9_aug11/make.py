"""
qm9_aug11 -- config generator for the conditional MOLECULE ladder.

    python configs/qm9_aug11/make.py             # write the arm configs + INDEX
    python configs/qm9_aug11/make.py --preflight # verify every referenced data file exists

Three arms, identical in everything but the number of distinct molecules the policy is
conditioned on:

    v1  1 molecule    -- the CONTROL. One condition, so every per-condition statistic
                         degrades to pooled and the run is unconditional in all but
                         plumbing. What it isolates is the cost of carrying the conditioner
                         and a scalarMLP flow head at all, with no condition diversity to
                         explain. If this arm does not converge, nothing above it will.
    v2  2 molecules   -- the smallest run where Z(c) has to separate. The first place a
                         conditional failure can show up as a per-condition spread rather
                         than a level error.
    v8  8 molecules   -- the N-molecule arm, plus held-out conditions for the
                         generalization check.

Arms share base.yaml entirely; only the four keys in build_config() vary. base.yaml is a
SNAPSHOT of mk_dev.yaml (2026-08-11) with the molecule-conditional layer folded in --
re-snapshot by hand to pick up mk_dev changes, deliberately, so this battery does not drift
under the user's live dev config.

=============================================================================
WHY 8 AND NOT 20
=============================================================================
The source set (eval_qm9_sg2_dataset.pt) has 20 molecules, but standardizing them to the
trainer's frame applies an improper (mirroring) transform to 12 of them -- a property of
the molecule's inertia eigenbasis, so it is all-or-nothing per molecule, never per replica.
A mirrored molecule's stored aunit_orientation cannot be repaired (a rotation vector cannot
encode a reflection), so its baked cell parameters no longer describe its crystal. Since
train_prior here uses bwd_sampling_mode: dataset -- it trains ON those parameters -- every
arm is built with --valid-only and the ladder tops out at the 8 molecules that survived.

The 12 mirrored molecules are not wasted: they are perfectly good CONDITIONS (the molecule
and its embedding agree with each other), and held-out evaluation only ever forward-samples,
so they become v8's test_molecules_path. That also makes the generalization check a real
one -- those molecules are absent from training entirely, not merely held-out replicas of
molecules the policy has already seen. eval_qm9_sg2 and test_eval_qm9_sg2 share all 20
molecules, so slicing THAT pair would have tested memorization of replicas, not transfer.

=============================================================================
REGENERATING THE DATA
=============================================================================
    python build_qm9_conditions.py --source D:\crystal_datasets\eval_qm9_sg2_dataset.pt \\
        --out-dir D:\crystal_datasets\conditional\priors --tag qm9_v8 \\
        --n-molecules 8 --valid-only

Anything that regenerates these files MUST preserve the frame fixed-point property that
script verifies, or the embedding stops describing the molecule the crystal is built from.
"""

import argparse
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))

from configs.generate_configs import overwrite_nested_dict  # noqa: E402

PRIORS = r'D:\crystal_datasets\conditional\priors'

# (name, tag suffix, data tag, held-out condition set or None)
TARGETS = [
    ('qm9_v1', 'qm9_v1', None),
    ('qm9_v2', 'qm9_v2', None),
    ('qm9_v8', 'qm9_v8', 'qm9_heldout'),
]


def build_config(base, name, data_tag, heldout):
    cfg = overwrite_nested_dict(yaml.safe_load(yaml.safe_dump(base)), {
        'run_name': name,
        'prior_path': rf'{PRIORS}\{data_tag}_prior.pt',
        'molecules_path': rf'{PRIORS}\{data_tag}_conditions.pt',
        'test_molecules_path': rf'{PRIORS}\{heldout}_conditions.pt' if heldout else None,
    })
    # anchor seeding follows the molecule domain rule and must track the arm's own prior
    cfg['buffers']['anchor_buffer']['seed_source'] = 'prior_dataset'
    return cfg


def assert_fresh(cfg, name):
    """Every arm trains its own phase 1.

    No phase1_exit checkpoint exists for this problem identity -- it is a new energy/prior
    combination -- and get_problem_definition keys identity on prior_path, so a checkpoint
    borrowed from another arm would hard-fail assert_problem_match. Asserted rather than
    trusted because a stale checkpoint_name silently retrains or silently cross-loads, and
    both are invisible in the results.
    """
    assert cfg['checkpoint_name'] is None, f'{name}: checkpoint_name must be null'
    assert cfg['prior_model_name'] is None, f'{name}: prior_model_name must be null'
    assert cfg['continue_from_checkpoint'] is False, f'{name}: continue_from_checkpoint must be false'


def assert_conditioner(cfg, name):
    """The embedding route, and only it."""
    assert cfg['embedding_conditioning'] is True, f'{name}: embedding_conditioning must be on'
    assert cfg['molecule_conditioning'] is False, f'{name}: live-GNN route must stay off'
    assert cfg['vector_conditioning'] is False, f'{name}: `c` route must stay off'
    assert cfg['embedding_conditioning_dim'] == 192, f'{name}: dim must match Mo3ENet 3x64'


def preflight(configs):
    missing = []
    for name, cfg in configs:
        for key in ('prior_path', 'molecules_path', 'test_molecules_path'):
            path = cfg.get(key)
            if path and not Path(path).exists():
                missing.append(f'  {name}.{key}: {path}')
    if missing:
        print('MISSING data files:\n' + '\n'.join(missing))
        return 1
    print(f'preflight OK: every data file referenced by {len(configs)} arms exists')
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preflight', action='store_true')
    args = ap.parse_args()

    base = yaml.safe_load((HERE / 'base.yaml').read_text())

    configs = []
    for name, data_tag, heldout in TARGETS:
        cfg = build_config(base, name, data_tag, heldout)
        assert_fresh(cfg, name)
        assert_conditioner(cfg, name)
        configs.append((name, cfg))

    if args.preflight:
        raise SystemExit(preflight(configs))

    rows = ['name\tmolecules_path\ttest_molecules_path']
    for name, cfg in configs:
        out = HERE / f'{name}.yaml'
        out.write_text(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False))
        rows.append(f"{name}\t{Path(cfg['molecules_path']).name}\t"
                    f"{Path(cfg['test_molecules_path']).name if cfg['test_molecules_path'] else '-'}")
        print(f'wrote {out}')
    (HERE / 'INDEX.tsv').write_text('\n'.join(rows) + '\n')
    print(f'wrote {HERE / "INDEX.tsv"}')


if __name__ == '__main__':
    main()
