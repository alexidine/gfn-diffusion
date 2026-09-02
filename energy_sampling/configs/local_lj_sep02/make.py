"""local_lj_sep02: LOCAL functional test rig for the prior-sampler fix and the
lj_coeff relocation. Not a battery -- a short, watchable equilibration run.

    python configs/local_lj_sep02/make.py

WHY IT CAN RUN LOCALLY. `mk_dev` hashes to `e01bd1`, identical to
`dev_elj_p2_cruise_elj-mipcas_sg2_zp1_elj_prior_dataset-T2.5-e01bd1_phase1_exit.pt`
in D:/crystal_datasets/gfn_checkpoints (stage `train_prior`, step_ind 2910,
train_T 10), and that checkpoint has its buffer sidecar. So the whole phase-2
entry path -- stub exit, snapshot_prior, rebuild_prior_by_churn, then real
equilibration with a live prior buffer -- runs on the dev box with no cluster.

THE TWO ARMS ARE THE SAME CONFIG, RUN DIFFERENTLY. That is the point:

  fresh    checkpoint_name = the phase-1 exit
           -> passes THROUGH train_prior, so snapshot_prior fires, prior_model
              exists, and prior_buffer_has_sampler must read 1 with nonzero
              prior_buffer_added / turnover.

  resume   checkpoint_name = this run's own _running.pt, written by `fresh`
           -> resumes INSIDE equilibration, skipping the stub. On TODAY's code
              prior_model is absent, _prior_churn_cycle returns at its guard,
              and the buffer freezes (added 0, evicted 0, turnover 0,
              prior_admit_rate NaN). That is the bug, reproduced locally in
              minutes instead of inferred from a 20k-step cluster run.
              With the fix it must read has_sampler 1 and keep churning.

So `resume` is a test that FAILS on the current code by construction, which is
the only kind worth writing for this defect.

DELIBERATELY SMALL. eval_period 100 with batch 400 at T=10: several evals inside
a few hundred steps, so the prior-buffer channels (added / evicted / expired /
turnover / has_sampler / admit_rate) all get multiple samples quickly. This is a
behaviour probe, not a convergence run -- nothing here should be read as a
quality result.

RUN IT (see reference_local_run_recipe -- the PATH python has no torch):

    $env:PYTHONPATH = "C:\\Users\\mikem\\Projects\\mxt_gfn\\mxtaltools;C:\\Users\\mikem\\Projects\\mxt_gfn\\gfn_diffusion"
    & "C:\\Users\\mikem\\venvs\\csd_mxt_gfn\\Scripts\\python.exe" train.py --config configs\\local_lj_sep02\\lj_fresh.yaml
"""
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
MK_DEV = HERE.parent / 'mk_dev.yaml'

CKPT_DIR = 'D:/crystal_datasets/gfn_checkpoints'
SEED_EXIT = ('dev_elj_p2_cruise_elj-mipcas_sg2_zp1_elj_prior_dataset'
             '-T2.5-e01bd1_phase1_exit.pt')
EXIT_STEP = 2910          # stage train_prior, from the checkpoint's modeller_state
STEPS = 600
BATCH = 400
EVAL_PERIOD = 100
BWD_HI = 0.93


def base():
    with MK_DEV.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def committed_energy_kwargs():
    """energy_config keys the COMMITTED MolecularCrystal.__init__ accepts.

    mk_dev is live and carries keys (prior_flow_path, lambda_mix) whose consumer
    is still an uncommitted edit, so generating from it can produce a config the
    committed code cannot construct -- `TypeError: unexpected keyword argument`
    at startup. Same failure that killed mipu_bwd_aug31's first submit.
    """
    import ast
    import subprocess
    for path in ('energy_sampling/energies/molecular_crystal.py',
                 'energies/molecular_crystal.py'):
        try:
            src = subprocess.run(['git', 'show', f'HEAD:{path}'], capture_output=True,
                                 text=True, check=True,
                                 cwd=str(HERE.parent.parent)).stdout
        except Exception:
            continue
        if not src.strip():
            continue
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.ClassDef) and node.name == 'MolecularCrystal':
                for fn in node.body:
                    if isinstance(fn, ast.FunctionDef) and fn.name == '__init__':
                        return {a.arg for a in fn.args.args} - {'self'}
    return None


def build(seed_name, tag):
    cfg = base()
    cfg['run_name'] = f'ljtest_{tag}'
    cfg['tag'] = 'ljtest'
    cfg['checkpoints_dir'] = CKPT_DIR
    cfg['checkpoint_name'] = seed_name
    cfg['load_weights_only'] = False
    cfg['continue_from_checkpoint'] = False
    cfg['prior_model_name'] = None

    cfg['epochs'] = EXIT_STEP + STEPS
    cfg['batch_size'] = BATCH
    cfg['max_batch_size'] = BATCH
    cfg['grow_batch_size'] = True
    cfg['batch_util_target'] = 0
    cfg['eval_period'] = EVAL_PERIOD
    cfg['eval_num_samples'] = 500
    cfg['figs_period'] = STEPS * 10          # no figures; this is a metrics probe
    cfg['traj_checkpoint'] = False           # T=10 locally: activations are cheap
    cfg['progress_gate']['level_window'] = 300

    lc = cfg['lr_control']
    lc['mode'] = 'fixed'
    lc['fixed_scale'] = 0.25
    lc['burn_in_steps'] = 50
    lc['burn_in_scale'] = 0.05
    lc['repeat_every'] = 0

    cfg['protocol'] = 'prod_eq'
    cfg['protocols']['prod_eq'] = {'stages': [
        {
            'name': 'train_prior',
            'train_mode': 'bwd',
            'bwd_sampling_mode': 'dataset',
            'flags': {'update_log_z': True, 'scramble_conditions': True},
            'loss_coeffs': {'bwd': {'mle': 1.0, 'tbc': 0.0, 'repeats': 1.0,
                                    'tb_z_source': 'persistent'}},
            # tick metric, not the gate: _progress_history is not checkpointed,
            # so a gate-keyed exit publishes 0 on the first post-resume eval and
            # wipes the restored streak
            'exit': [{'metric': 'bwd/mle', 'above': -1e9, 'patience': 1}],
            # THE action under test in job 1 -- it is the only producer of
            # prior_model, and a resume skips it
            'on_exit': ['snapshot_prior'],
        },
        {
            'name': 'equilibration',
            'train_mode': 'fused',
            'bwd_sampling_mode': 'prior',
            'flags': {'update_log_z': True, 'buffers_active': True,
                      'z_calibration': True},
            'on_enter': ['rebuild_prior_by_churn', 'bootstrap_z:train_conditioner'],
            'fracs': {'fwd': 0.05, 'bwd': BWD_HI, 'replay': 0.02},
            'min_fracs': {'fwd': 0.02, 'bwd': 0.02, 'replay': 0.02},
            'deactivate_threshold': 0.01,
            'loss_coeffs': {
                'fwd': {'tb': 1.0, 'freeze_policy': 1.0},
                'bwd': {'tb': 1.0, 'beta': 80},
                'replay': {'tb': 1.0, 'beta': 80},
            },
            'balance': {
                'kind': 'ratio', 'pinned': {'fwd': 0.05},
                'metrics': {'replay': 'fwd/over_coverage',
                            'bwd': 'bwd/relative_under_wcen'},
                'numerator': 'replay', 'setpoint': 5.0, 'gain': 0.05,
                'max_step': 0.05,
                'bounds': {'replay': [0.02, BWD_HI], 'bwd': [0.02, BWD_HI]},
                'converge_floor': 1.0,
            },
        },
    ]}

    accepted = committed_energy_kwargs()
    if accepted is not None:
        for k in sorted(set(cfg['energy_config']) - accepted):
            del cfg['energy_config'][k]
    return cfg


def main():
    arms = {
        # passes THROUGH the stub: snapshot_prior fires, sampler exists
        'lj_fresh': build(SEED_EXIT, 'fresh'),
        # resumes INSIDE equilibration: reproduces the frozen prior buffer.
        # The placeholder is replaced by hand (or by the runner) with the
        # _running.pt that lj_fresh writes -- it cannot exist until then.
        'lj_resume': build('RESUME_RUNNING_PT_PLACEHOLDER', 'fresh'),
    }
    for name, cfg in arms.items():
        with (HERE / f'{name}.yaml').open('w', encoding='utf-8') as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
        print(f'{name}.yaml  run_name={cfg["run_name"]}  epochs={cfg["epochs"]} '
              f'batch={cfg["batch_size"]} eval={cfg["eval_period"]}')
    print(f'\nseed: {CKPT_DIR}/{SEED_EXIT}')
    print('lj_resume needs its checkpoint_name filled in with the _running.pt '
          'that lj_fresh writes.')


if __name__ == '__main__':
    main()
