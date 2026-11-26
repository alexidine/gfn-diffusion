"""pool4_nov26b: weight decay on MLE, on pool4_nov26's arms (configs/pool4_nov26/make.py).

  wd5e2       as pool4_nov26's mle4_ctrl (mle4's final weights, MLE_WARM_STEPS more MLE steps) with decoupled weight
              decay 5e-2 (AdamW on the policy, none on the Z head): compare with mle4_ctrl
  wd1e1       wd5e2 at 1e-1
  smoke_*     `smoke: true`, SMOKE_STEPS MLE steps, one per arm, at the first array indices.
Every arm stops at its MLE stage's exit (snapshot:phase1_exit)."""
import copy
import importlib.util
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('n26', HERE.parent / 'pool4_nov26' / 'make.py')
n26 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(n26)
p4 = n26.p4

TAG = 'pool4_nov26b'
SMOKE_STEPS = n26.SMOKE_STEPS


def decayed(name, steps, wd):
    cfg = n26.warm(name, steps)
    cfg.update(use_weight_decay=True, weight_decay=wd, weight_decay_kind='decoupled', tag=TAG)
    return cfg


ARMS = {
    'wd5e2': lambda s: decayed('wd5e2', s, 5.0e-2),
    'wd1e1': lambda s: decayed('wd1e1', s, 1.0e-1),
}
STEPS = {'wd5e2': n26.MLE_WARM_STEPS, 'wd1e1': n26.MLE_WARM_STEPS}


def configs():
    out = {}
    for name, build in ARMS.items():
        sm = build(SMOKE_STEPS)
        sm.update(run_name=f'smoke_{name}', smoke=True, eval_period=100, archive_period=10 ** 6)
        out[sm['run_name']] = sm
    for name, build in ARMS.items():
        out[name] = build(STEPS[name])
        out[name]['run_name'] = name
    return out


if __name__ == '__main__':
    cfgs = configs()
    for name, cfg in cfgs.items():
        assert cfg['run_name'] == name and cfg['tag'] == TAG
        (HERE / f'{name}.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8', newline='\n')
    text = p4.sbatch(list(cfgs)).replace(p4.TAG, TAG)
    (HERE / f'submit_{TAG}.sbatch').write_text(text, encoding='utf-8', newline='\n')
    full = [n for n in cfgs if not n.startswith('smoke_')]
    for name in full:
        cfg = copy.deepcopy(cfgs[name])
        cfg.update(continue_from_checkpoint=True, checkpoint_name=None, load_weights_only=False)
        (HERE / f'{name}_resume.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8', newline='\n')
    (HERE / f'submit_{TAG}_resume.sbatch').write_text(p4.sbatch([f'{n}_resume' for n in full]).replace(p4.TAG, TAG),
                                                      encoding='utf-8', newline='\n')
    print(' '.join(cfgs))
