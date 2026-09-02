r"""
Build the lambda-anneal arm from the qm9c_lam003 base, adopting mk_dev's
rewired var_conditioning stage: pooled VarGrad only (+ the Z sidecar), fracs
pinned 50:50, and lambda_mix ramped by the stage's own anneal_coeffs under a
loss gate.

WHY 0.01 IS THE START AND NOT 0. The ramp is multiplicative (`val <- val/rate`),
so it cannot leave exactly zero. And it should not start near zero anyway:
measured on 32000 policy draws x the real ELJ (scratch lambda_scan2.py), the
physical leg's within-condition spread is under a quarter of the flow leg's for
all lambda < 0.03, so the whole region below that costs wall clock and moves the
target almost not at all. lambda = 0.003, where the previous ladder sat, puts the
physical leg at 13% of the flow leg.

    python configs/qm9c_anneal/make.py
"""
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIGS = os.path.dirname(HERE)
REPO = os.path.dirname(CONFIGS)
for p in (REPO, os.path.dirname(REPO), os.path.join(os.path.dirname(REPO), 'mxtaltools')):
    if p not in sys.path:
        sys.path.insert(0, p)

BASE = os.path.join(CONFIGS, 'qm9c_lam003.yaml')
MK_DEV = os.path.join(CONFIGS, 'mk_dev.yaml')
OUT = os.path.join(CONFIGS, 'qm9c_anneal.yaml')

LAMBDA_START = 0.01
EPOCHS = 40000


def stage_named(cfg, name):
    for prot in cfg['protocols'].values():
        for st in prot['stages']:
            if st['name'] == name:
                return st
    raise KeyError(name)


def main():
    cfg = yaml.safe_load(open(BASE))
    mk = yaml.safe_load(open(MK_DEV))
    ref = stage_named(mk, 'var_conditioning')      # the rewired stage, single source

    cfg['run_name'] = 'qm9c_anneal'
    cfg['epochs'] = EPOCHS

    ec = cfg['energy_config']
    ec['lambda_mix'] = LAMBDA_START
    assert ec.get('prior_flow_path'), 'the anneal needs a fitted prior flow'
    # the clip is a nonlinear rescale of the PHYSICAL energy; applied to a
    # mixture it would make the lambda=0 endpoint something other than the flow,
    # and MolecularCrystal refuses the combination outright
    ec['reward_range'] = None

    # base pooled keys must EXIST (mk_dev owns the schema) but stay 0 in the base
    # block: runs_grouped_vargrad is not scoped by train_mode, so a nonzero base
    # pooled_vg flips condition_block_m 0 -> 2 in every bwd stage, train_prior
    # included. The stage below turns it on where it is actually computed.
    for k, v in (('pooled_vg', 0.0), ('pooled_beta', 40.0), ('pooled_ratio', 0.5)):
        cfg['fwd_loss_coeffs'][k] = v

    st = stage_named(cfg, 'var_conditioning')
    for key in ('fracs', 'loss_coeffs', 'balance', 'hot_lr_sensor'):
        st[key] = ref[key]
    st.pop('min_fracs', None)

    with open(OUT, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False, width=1000)

    # LOADING IS THE TEST, not generating. Re-read from disk and run the
    # invariants; configs/generate.py asserts on errors for the same reason.
    import config_invariants as ci
    back = yaml.safe_load(open(OUT))
    errs = ci.errors(back)
    for v in ci.check(back):
        print('  ', v)
    assert not errs, f'{len(errs)} invariant ERROR(s) in the generated config'

    got = stage_named(back, 'var_conditioning')
    assert got['balance']['kind'] == 'lexicographic'
    assert got['balance']['anneal_coeffs']['lambda_mix']['target'] == 1.0
    assert got['fracs'] == {'fwd': 0.5, 'bwd': 0.5, 'replay': 0.0}
    assert got['loss_coeffs']['fwd']['pooled_vg'] == 1.0
    assert got['loss_coeffs']['fwd']['vg_lb'] == 0.0
    assert got['loss_coeffs']['bwd']['vg_lb'] == 0.0
    assert got['loss_coeffs']['fwd']['emp_z'] == 1.0
    assert back['energy_config']['lambda_mix'] == LAMBDA_START
    assert back['energy_config']['reward_range'] is None
    # the pooled term is the ONLY thing arming the blocked backward draw now
    assert ci._runs_vargrad(back, got, 'bwd'), 'blocked backward draw would be OFF'
    for prot in back['protocols'].values():
        for s in prot['stages']:
            if s.get('train_mode') == 'bwd':
                assert not ci._runs_vargrad(back, s, 'bwd'), \
                    f"stage {s['name']}: pooled_vg leaked into a bwd stage"
    print(f'wrote {OUT}  (lambda_mix starts at {LAMBDA_START}, epochs {EPOCHS})')


if __name__ == '__main__':
    main()
