"""dose_sep16 -- the replay-dose ladder on mip: does a FRESHER buffer hold the
high-Z state that N=20 only visits?

    python configs/dose_sep16/make.py

THE HYPOTHESIS (owner + 2026-09-16 readout of flk_sep14 and the four p12 arms).
Across the production arms the replay dose D = N * (lr/1e-4) * (w/0.1) * (1000/B)
orders everything: memorisation (replay/resid_vs_intake 0.70/0.75/0.81/0.90 at
D = 15.6/7.8/2.1/1.0 for mip/neh/nehu/mipu), whether log Z climbs or sits flat
with transient peaks, and whether the ratchet holds replay at the floor. mipu,
at D ~ 1 and memorisation 0.90, is the only arm that looks healthy. mip's
excursions to 31-32 were the policy broadening on a burst of fresh rollouts and
then memorising the same rows at 16x mipu's rate; pinned at 0.3 replay it holds
29.8 (flk14_pin30). Rare rollouts were bought for the MLIP routes; on ELJ a
rollout is nearly free, so N=20 buys nothing there and costs the freshness.

THE LADDER, all replay PINNED at 0.3 unless stated (point bounds, as flk_sep14),
all off the SAME 160k archive flk_sep14 used (read from its SEED.txt if present,
else the parent's newest archive, recorded the same way):

  n20          D ~ 47   the flk14_pin30 shape, re-run to 25k steps as control
  n5           D ~ 12
  n1           D ~ 2.3  fresh replay every step at share 0.3
  n1_lr05      D ~ 1.2  n1 at half the rate, applied through seed_lr because a
               resumed controller keeps the parent's promoted scale (see _rate)
  n1_w15       D ~ 1.2  n1 at share 0.15 -- the same dose by the other lever

fwd_rollout_every 1 keeps the cadenced code path (fill pins Z on every step);
0 would take the legacy every-step path and is deliberately not used, so the
only thing that changes down the ladder is N.

WALL 12 h. At N=20 mip runs 2.6 s/step; at N=1 the forward rollout is on every
step but ELJ is cheap, so budget ~3.5 s/step -> ~12k steps at N=1, ~16k at N=20.
Not the 25k a full buffer turnover wants, but the replay buffer itself turns over
in tau = 120 steps, so the dose effect is visible in a few thousand.

WHAT TO READ: fwd/log_Z_learned level and slope over the last 4k, its sd;
replay/resid_vs_intake (mipu's 0.90 is the target; the 1/e line is 0.37);
fwd/tb_err and eval_fwd/tb_err; fwd/emp_z minus learned (the ELBO gap);
bwd/logw_std_within; Excess Energy Nats Mean (breadth); protocol/gr_held (must
stay ~0 under a pin). The n1_lr05 vs n1_w15 pair says whether dose is the whole
story or the share has a separate role.
"""
import copy
import importlib.util
import pathlib
import subprocess
import sys

import yaml

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
ES = ROOT.parent
BASE = ROOT / 'prod_sep12' / 'p12_mip_lr1.yaml'
PARENT = 'p12_mip_lr1'
PLACEHOLDER = 'WARM_CHECKPOINT_PLACEHOLDER'
PRIOR_PLACEHOLDER = 'PRIOR_MODEL_PLACEHOLDER'
TAG = 'dose16'
WALL = '12:00:00'
EXECUTED = ('configs/mk_dev.yaml', 'train.py', 'protocol.py', 'buffer.py',
            'gflownet_losses.py', 'checkpointing.py', 'utils.py', 'config_invariants.py',
            'models/gfn.py', 'energies/molecular_crystal.py')

_spec = importlib.util.spec_from_file_location('flkmake', ROOT / 'flk_sep14' / 'make.py')
flk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(flk)
_eq, _pin = flk._eq, flk._pin


def _every(n):
    def f(cfg):
        _eq(cfg)['fwd_rollout_every'] = int(n)
    return f


def _rate(scale):
    # THE RATE ON A RESUME IS seed_lr, NOT fixed_scale. The controller comes back
    # from the parent's checkpoint in CRUISE with the parent's promoted scale
    # (1.0), and fixed mode stamps `fixed_scale` only at the end of a burn-in that
    # a resume never repeats -- so a config fixed_scale 0.5 would run at 1.0.
    # `_apply_lrs` computes lr = base x scale where every `auto` lr_* key resolves
    # to seed_lr, so halving seed_lr halves the applied rate whatever was restored.
    def f(cfg):
        cfg['lr_control']['seed_lr'] = 1.25e-4 * float(scale)
    return f


ARMS = {
    'n20':      [lambda c: _pin(c, 0.3), _every(20)],
    'n5':       [lambda c: _pin(c, 0.3), _every(5)],
    'n1':       [lambda c: _pin(c, 0.3), _every(1)],
    'n1_lr05':  [lambda c: _pin(c, 0.3), _every(1), _rate(0.5)],
    'n1_w15':   [lambda c: _pin(c, 0.15), _every(1)],
}
EXPECT = {'n20': (20, 0.3, 1.0), 'n5': (5, 0.3, 1.0), 'n1': (1, 0.3, 1.0),
          'n1_lr05': (1, 0.3, 0.5), 'n1_w15': (1, 0.15, 1.0)}


def dirty_files():
    out = subprocess.run(['git', 'status', '--porcelain', '--'] + [str(ES / p) for p in EXECUTED],
                         capture_output=True, text=True, cwd=str(ES), check=True).stdout
    return [line[3:] for line in out.splitlines() if line.strip()]


def dose(n, scale, w, batch=1600):
    return n * (1.25e-4 * scale / 1e-4) * (w / 0.1) * (1000.0 / batch)


def build():
    base = yaml.safe_load(BASE.read_text(encoding='utf-8'))
    out = {}
    for arm, deltas in ARMS.items():
        cfg = copy.deepcopy(base)
        name = TAG + '_' + arm
        cfg['run_name'] = name
        cfg['tag'] = TAG
        cfg['checkpoint_name'] = PLACEHOLDER
        cfg['prior_model_name'] = PRIOR_PLACEHOLDER
        cfg['load_weights_only'] = False
        cfg['continue_from_checkpoint'] = False
        for d in deltas:
            d(cfg)
        check(cfg, name, arm)
        out[name] = cfg
    return out


def check(cfg, name, arm):
    n, w, scale = EXPECT[arm]
    eq = _eq(cfg)
    assert cfg['checkpoint_name'] == PLACEHOLDER and cfg['prior_model_name'] == PRIOR_PLACEHOLDER, name
    assert cfg['load_weights_only'] is False and cfg['epochs'] >= 500_000, name
    assert eq['fwd_rollout_every'] == n and n >= 1, name + ': N must stay on the cadenced path (>= 1)'
    assert eq['flags']['z_calibration'] is False and float(cfg['z_calibration']['fill_threshold']) > 0, name
    b = eq['balance']; bwd = round(1 - w, 3)
    assert eq['fracs'] == {'fwd': 0.0, 'bwd': bwd, 'replay': w}, name
    assert b['bounds'] == {'bwd': [bwd, bwd], 'replay': [w, w]}, name + ': bounds must be a point'
    lc = cfg['lr_control']
    assert lc['mode'] == 'fixed' and lc['fixed_scale'] == lc['burn_in_scale'] == 1.0, name + ': the scale is inherited; the rate moves through seed_lr'
    assert abs(lc['seed_lr'] - 1.25e-4 * scale) < 1e-12, name
    assert all(cfg[k] == 'auto' for k in ('lr_policy', 'lr_back', 'lr_replay', 'lr_fused')), name + ': seed_lr only reaches auto keys'
    assert cfg['buffers']['prior_buffer']['source'] == 'prior_model', name
    assert cfg['buffers']['replay_buffer']['churn_rate'] == 0 and cfg['buffers']['replay_buffer']['mean_residence_steps'] == 120, name
    for s in cfg['protocols']['unconditional_tb']['stages']:
        sensor = s.get('hot_lr_sensor')
        if isinstance(sensor, dict):
            assert sensor.get('action', 'report') == 'report', name


# The seed block reads flk_sep14's SEED.txt first so the two batteries share one
# archive; otherwise it records the parent's newest archive exactly as flk did.
SBATCH = flk.SBATCH.replace('#SBATCH --job-name=flk14', '#SBATCH --job-name=dose16') \
    .replace('configs/flk_sep14/joblogs/%x_%A_%a.out', 'configs/dose_sep16/joblogs/%x_%A_%a.out') \
    .replace('# flk_sep14: seven knobs off one frozen p12_mip_lr1 archive.',
             '# dose_sep16: the replay-dose ladder off the SAME frozen p12_mip_lr1 archive flk_sep14 used.') \
    .replace('ARMS=${{WORKDIR}}/configs/flk_sep14', 'ARMS=${{WORKDIR}}/configs/dose_sep16') \
    .replace('SEED_FILE=${{LOGS}}/SEED.txt',
             'SEED_FILE=${{LOGS}}/SEED.txt\nFLK_SEED=${{WORKDIR}}/configs/flk_sep14/joblogs/SEED.txt\n'
             'if [ ! -s ${{SEED_FILE}} ] && [ -s ${{FLK_SEED}} ]; then cp ${{FLK_SEED}} ${{SEED_FILE}}; echo "  seed inherited from flk_sep14: $(cat ${{SEED_FILE}})"; fi')


def main(argv):
    dirty = dirty_files()
    if dirty and '--allow-dirty' not in argv:
        sys.exit('REFUSING: uncommitted files the arms execute:\n  ' + '\n  '.join(dirty) +
                 '\nBuild from a clean worktree, or pass --allow-dirty for a LOCAL build.')
    arms = build()
    logs = HERE / 'joblogs'
    logs.mkdir(exist_ok=True)
    (logs / '.gitkeep').write_text('SLURM cannot create --output; SEED.txt is written here at launch\n', encoding='utf-8')
    for name, cfg in arms.items():
        with (HERE / (name + '.yaml')).open('w', encoding='utf-8') as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    with (HERE / 'INDEX.tsv').open('w', encoding='utf-8', newline='\n') as f:
        f.write('arm\twarm_src\n')
        for name in arms:
            f.write('%s\t%s\n' % (name, PARENT))
    with (HERE / 'submit_dose_sep16.sbatch').open('w', encoding='utf-8', newline='\n') as f:
        f.write(SBATCH.format(wall=WALL, last=len(arms) - 1, placeholder=PLACEHOLDER,
                              prior_placeholder=PRIOR_PLACEHOLDER))
    for name, cfg in arms.items():
        n, w, scale = EXPECT[name[len(TAG) + 1:]]
        print('%-16s N=%-3d replay=%.2f scale=%.2f  dose~%5.1f' % (name, n, w, scale, dose(n, scale, w)))


if __name__ == '__main__':
    main(sys.argv[1:])
