"""pool4_nov26: effective-temperature MLE and capacity, MLE only, on pool4_nov24's base (configs/pool4_nov24/make.py).

  mle4_ctrl   mle4's final weights, the MLE stage MLE_WARM_STEPS more steps, no temperature (the control for tcond_*)
  tcond_c1    as mle4_ctrl with temperature_conditioning: each live kick labelled T_eff = kappa_per_atom x atoms x rho^2 / d
              (KAPPA_PER_ATOM), today's kick radii; the temperature column is added to mle4's weights at zero weight
  tcond_c2    tcond_c1 with the kick radii at HOT_LOG_RANGE (0.25-0.8 A, T_eff about 1-12 for a 15-atom molecule)
  cap_policy  fresh MLE as pool4_nov24's mle_live, with a larger flow (tf_width, tf_layers): compare with mle_live
  cap_encoder fresh MLE as mle_live, with a larger molecule encoder (atom_node_dim, atom_convs): compare with mle_live
  smoke_*     `smoke: true`, SMOKE_STEPS MLE steps, one per arm, at the first array indices.
Every arm stops at its MLE stage's exit (snapshot:phase1_exit)."""
import copy
import importlib.util
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('p4', HERE.parent / 'pool4_nov24' / 'make.py')
p4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p4)

TAG = 'pool4_nov26'
MLE_WARM_STEPS, MLE_FRESH_STEPS, SMOKE_STEPS = 40000, 150000, 300
KAPPA_PER_ATOM = 14.9            # energy units per molecule per A^2 per atom (kick-stiffness scan, nov26)
LOG_T_RANGE = [-3.0, 1.5]        # log10 kT the temperature column is scaled over
HOT_LOG_RANGE = [-0.6, -0.1]     # log10 A: kick radii 0.25-0.8 A


def mle_only(cfg, steps):
    mle = p4._mle(cfg)
    mle.pop('skip_if', None)
    mle.update(max_steps=steps, on_exit=['snapshot:phase1_exit', 'stop'])
    cfg['protocols'][p4.uni.PROTOCOL]['stages'] = [mle]
    cfg['epochs'] = steps + p4.uni.LOOP_MARGIN
    cfg['tag'] = TAG
    return cfg


def warm(name, steps, tcond=False, radii=None):
    cfg = mle_only(p4.from_mle4(name, p4.base(mle_steps=steps)), steps)
    if tcond:
        cfg['temperature_conditioning'] = True
        cfg['noised_prior']['kick_kappa_per_atom'] = KAPPA_PER_ATOM
        cfg['energy_config']['log_temperature_range'] = list(LOG_T_RANGE)
        cfg['warm_start_ignore_problem_keys'] = ['temp_cond', 'energy_config.log_temperature_range']
    if radii is not None:
        cfg['noised_prior']['live_log_range'] = list(radii)
    return cfg


def fresh(name, steps, nf):
    cfg = p4.mle_live(name, mle_steps=steps)
    cfg['model']['nf'].update(nf)
    cfg['tag'] = TAG
    return cfg


ARMS = {
    'mle4_ctrl': lambda s: warm('mle4_ctrl', s),
    'tcond_c1': lambda s: warm('tcond_c1', s, tcond=True),
    'tcond_c2': lambda s: warm('tcond_c2', s, tcond=True, radii=HOT_LOG_RANGE),
    'cap_policy': lambda s: fresh('cap_policy', s, dict(tf_width=768, tf_layers=12)),
    'cap_encoder': lambda s: fresh('cap_encoder', s, dict(atom_node_dim=384, atom_convs=6)),
}
STEPS = {'mle4_ctrl': MLE_WARM_STEPS, 'tcond_c1': MLE_WARM_STEPS, 'tcond_c2': MLE_WARM_STEPS,
         'cap_policy': MLE_FRESH_STEPS, 'cap_encoder': MLE_FRESH_STEPS}


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
