"""acr_nov25: unconditional acridine (P2_1/c, Z' = 1) on the unified store route under an MLIP, one arm per model.

  base      pool4_nov24's base (the committed mk_dev through unified_nov23: store on, MLE on the live physical kernel
            over anchors, then fused TB with bwd and replay 50:50 and the forward sidecar), with the problem set to one
            molecule and one space group: energy_function and mlip_path from MODELS, the MACE route's D3 where the
            model needs it (energy_config.mlip_dispersion), space_groups [14], no held-out molecule file, no Z(sg)
            panel (one group). The TB stage's LR ramps over RAMP_STEPS (owner rule 2026-11-25).
  prior     PRIORS[model]: the model's band set (MXtalTools configs/crystal_searches/acr_mlip_nov19/band.py: states
            within 10 kT of the lowest minimum, 0.1 A ball net, with each state's image under the molecule's C2),
            built by data_processing/build_noised_prior.py --basins with no kicked children (--k 0), each anchor again
            as its own radius-0 child (--self_children: the TB stage draws its backward rows from children), at the
            energies the search recorded (--stored_energies: no MLIP pass; no forces) (build_prior.sbatch); its
            conditions file is the lowest training anchor (molecule_split_files.py).
  smoke     `smoke: true`, SMOKE_MLE_STEPS MLE then SMOKE_TB_STEPS TB steps.
  sbatch    submit_acr_nov25.sbatch: one arm per array index, the smokes first (0-4), then the full arms (5-9).

usage: python configs/acr_nov25/make.py                       (cluster arms and the sbatch file)
       python configs/acr_nov25/make.py --local DIR --prior P --conditions C [--model M]
                                                              (a dev-box config: build test or local smoke)"""
import argparse
import copy
import importlib.util
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('pool4_nov24', HERE.parent / 'pool4_nov24' / 'make.py')
p4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p4)
uni = p4.uni

TAG = 'acr_nov25'
DATA = '/scratch/mk8347/data/crystal_datasets/acridine'
PRIOR_DIR = f'{DATA}/priors/{TAG}'
MODELS = {   # energy_function, cluster mlip_path, dev-box mlip_path, energy_config.mlip_dispersion
    'macecspl': ('mace', '/scratch/mk8347/models/day_models/mace_csp_25.2_large.model',
                 'D:/crystal_datasets/day_models/mace_csp_25.2_large.model', {'xc': 'pbe', 'damping': 'bj'}),
    'maceoff24m': ('mace', '/scratch/mk8347/models/MACE-OFF24_medium.model',
                   'D:/crystal_datasets/MACE-OFF24_medium.model', None),
    'umas1p1': ('uma', '/scratch/mk8347/models/uma/uma-s-1p1.pt', 'D:/crystal_datasets/uma-s-1p1.pt', None),
    'umas1p2': ('uma', '/scratch/mk8347/models/uma/uma-s-1p2p1.pt', 'D:/crystal_datasets/uma-s-1p2p1.pt', None),
    'esens': ('uma', '/scratch/mk8347/models/uma/esen_s.pt', 'D:/crystal_datasets/esen_s.pt', None),
}
RAM_GB, MEM_GB = 64, 128      # buffers.store.ram_gb (one molecule: a 0.45M-row prior and its replay) and #SBATCH --mem
PYEXTRA = {'macecspl': '/scratch/mk8347/pylibs_dftd', 'umas1p2': '/scratch/mk8347/pylibs_fc216'}   # PYTHONPATH first
#: the MLIP density guard (energy_config.mlip_max_packing_coeff, read with implausible_energy_below), every arm: a cell
#: denser than MACE_MAX_PACKING is not handed to the MLIP and is scored |CLASH_BELOW| kJ/mol per molecule. An early
#: policy draws cells at packing ~2.4 whose MACE energy (~6e9) moves by ~4e4 with the batch it is scored in, and under
#: UMA-s 1.1 one such rollout asked 11 GB for one crystal (local smokes, 2026-11-25); the prior's rows lie in packing
#: 0.55-0.95
MACE_MAX_PACKING, CLASH_BELOW = 1.0, -1000.0
MLE_STEPS, TB_STEPS = 50000, 100000      # not calibrated for one molecule: the smoke and the first run's curves decide
SMOKE_MLE_STEPS, SMOKE_TB_STEPS = 300, 600
RAMP_STEPS = p4.RAMP_STEPS


def prior_files(model, prior_dir=PRIOR_DIR):
    return {'prior': f'{prior_dir}/acr_{model}_k0.pt', 'conditions': f'{prior_dir}/acr_{model}_k0_conditions.pt'}


def arm(model, files, mle_steps=MLE_STEPS, tb_steps=TB_STEPS, local=False):
    ef, path, local_path, disp = MODELS[model]
    cfg = p4.base(tb_steps=tb_steps, mle_steps=mle_steps)
    assert cfg['energy_function'] == 'elj' and cfg['mlip_path'] is None, (cfg['energy_function'], cfg['mlip_path'])
    assert 'mlip_dispersion' in cfg['energy_config'], 'the committed mk_dev lacks energy_config.mlip_dispersion'
    cfg.update(tag=TAG, run_name=model, energy_function=ef, mlip_path=local_path if local else path,
               prior_path=files['prior'], molecules_path=files['conditions'], test_molecules_path=None,
               space_groups=[14])
    cfg['energy_config']['mlip_dispersion'] = copy.deepcopy(disp)
    cfg['buffers']['store']['ram_gb'] = RAM_GB
    cfg['energy_config'].update(mlip_max_packing_coeff=MACE_MAX_PACKING, implausible_energy_below=CLASH_BELOW)
    if ef == 'uma':   # on, the backward through UMA raises CheckpointError: the prior build's forces (2026-11-25), as
        cfg['energy_config']['mlip_activation_checkpointing'] = False   # the searches' relaxations (oct27)
    cfg['zsg_panel'] = {'enabled': False, 'molecules': 64}
    tb = p4._tb(cfg)
    assert tb['lr_schedule'] == [{'to': 0.2, 'steps': 0, 'kind': 'hold'}], tb['lr_schedule']
    tb['lr_schedule'] = p4.ramp(0.2, RAMP_STEPS)
    return cfg


def smoke(cfg):
    cfg = copy.deepcopy(cfg)
    cfg.update(run_name=f"smoke_{cfg['run_name']}", smoke=True, eval_period=100, archive_period=10 ** 6,
               epochs=SMOKE_MLE_STEPS + SMOKE_TB_STEPS + uni.LOOP_MARGIN)
    p4._mle(cfg)['max_steps'] = SMOKE_MLE_STEPS
    p4._tb(cfg)['max_steps'] = SMOKE_TB_STEPS
    return cfg


def sbatch(arms):
    """submit_acr_nov25.sbatch from unified_nov23's template: one arm per array index (smokes first), no held-out
    molecule file, an arm's model's extra PYTHONPATH (PYEXTRA: torch-dftd for MACE-CSP, fairchem 2.16 for UMA-s 1.2)."""
    text = (HERE.parent / 'unified_nov23' / 'submit_template.sbatch').read_text(encoding='utf-8')
    text = (text.replace('@TAG@', TAG).replace('@ARMS@', ' '.join(arms)).replace('@NARMS@', str(len(arms) - 1))
            .replace('@RAM_GB@', str(MEM_GB)))
    old = '    for KEY in prior_path molecules_path test_molecules_path; do'
    assert text.count(old) == 1
    text = text.replace(old, '    for KEY in prior_path molecules_path; do')
    old = 'GROUP="${ARM_LIST[${SLURM_ARRAY_TASK_ID:?array task id}]}"'
    assert text.count(old) == 1
    cases = ''.join(f'    *{m}) PYEXTRA={p} ;;\n' for m, p in PYEXTRA.items())
    text = text.replace(old, old + "\ncase ${GROUP} in   # the model's extra PYTHONPATH (make.py PYEXTRA)\n" + cases
                        + '    *) PYEXTRA= ;;\nesac')
    old = 'export PYTHONPATH=${PROJECT_ROOT}/MXtalTools'
    assert text.count(old) == 1
    text = text.replace(old, 'export PYTHONPATH=${PYEXTRA:+${PYEXTRA}:}${PROJECT_ROOT}/MXtalTools')
    return text


def write(name, cfg, out=HERE):
    (Path(out) / f'{name}.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8', newline='\n')
    print(Path(out) / f'{name}.yaml')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--local', default=None, help='write one dev-box config into this directory')
    ap.add_argument('--prior', default=None)
    ap.add_argument('--conditions', default=None)
    ap.add_argument('--model', default='macecspl', choices=list(MODELS))
    ap.add_argument('--ckdir', default=None)
    a = ap.parse_args()
    if a.local:
        out = Path(a.local)
        out.mkdir(parents=True, exist_ok=True)
        c = smoke(arm(a.model, {'prior': a.prior, 'conditions': a.conditions}, local=True))
        uni.localize(c, a.ckdir or str(out / 'checkpoints'), eval_samples=500, eval_period=100)
        c['zsg_panel'] = {'enabled': False, 'molecules': 8}
        write(f'local_{a.model}', c, out)
    else:
        for m in MODELS:
            write(m, arm(m, prior_files(m)))
            write(f'smoke_{m}', smoke(arm(m, prior_files(m))))
        arms = [f'smoke_{m}' for m in MODELS] + list(MODELS)
        (HERE / f'submit_{TAG}.sbatch').write_text(sbatch(arms), encoding='utf-8', newline='\n')
        print(HERE / f'submit_{TAG}.sbatch', ': indices', ', '.join(f'{i} {a}' for i, a in enumerate(arms)))
