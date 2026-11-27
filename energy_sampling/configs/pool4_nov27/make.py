"""pool4_nov27: convergence at T 10, the temperature-first runs continued, greedier TB at T 2.5, and a large model
(configs/pool4_nov24, pool4_nov26 and pool4_nov26b make.py; owner go 2026-11-27).

Pathwise first (submit_pool4_nov27_pw.sbatch, its own array so it can be submitted ahead of the rest; smokes first): TB
at T 2.5 from mle4 as tb_ctrl, plus the pathwise reverse-KL term (gflownet_losses.py::pathwise_term) on PW_ROWS rows per
step per branch, each row's parameter gradient capped at PW_ROW_CAP x the rows' median, pathwise_entropy 1 (reverse KL),
no x clip, each branch's term scaled every step to pathwise_norm_frac x the norm of the step's other gradient on the
policy parameters (read pathwise/<branch>_frac). Compare each with tb_ctrl:
  pw_on           on-policy only: fwd, fresh draws at fresh conditions every fused step, fraction PW_FRAC
  pw_off          off-policy only: bwd (stored children) and replay (stored on-policy rows), each row regenerated
                  through the flow, fraction PW_FRAC per branch
  pw_both         pw_on and pw_off together, PW_FRAC per branch
  pw_both_hi      pw_both at PW_FRAC_HI per branch

Now (submit_pool4_nov27.sbatch; smokes first, then the arms in this order):
  mle_T10         mle4's weights, MLE_WARM_STEPS of MLE at energy_config.temperature 10 (density_coeff x T / 2.5, as
                  pool4_nov24 tb_m4_T10), each kick drawn as a Gaussian at T 10 (noised_prior.live_temperature_range
                  [10, 10], KAPPA_PER_ATOM; unlabelled: no temperature column). Compare with pool4_nov26 mle4_ctrl
  tcond_t_cont    pool4_nov26b tcond_t continued: weights-only from its phase1_exit, MLE_WARM_STEPS more of the same MLE
  tcond_fresh     fresh MLE as pool4_nov24 mle_live with tcond_t's temperature-first kicks over [2.5, 10] and its
                  temperature column, MLE_FRESH_STEPS. Compare with mle_live and tcond_t
  mt_T10          TB at T 10 (tb_m4_T10's settings) from tcond_t's phase1_exit, the policy conditioned on T 10. Compare
                  with pool4_nov24 tb_m4_T10 (TB at T 10 from the T-blind mle4)
  tb_ctrl         pool4_nov24 tb_m4 with the LR ramped (pool4_nov24 ramp): the reference for the tb_* arms below, and a
                  second seed of tb_m4
  tb_alpha25      tb_ctrl with fwd z_level_alpha 0.25: Z(c) fitted a quarter of the way from J_F(c) toward J_B(c)
  tb_alpha50      tb_ctrl at z_level_alpha 0.5
  tb_knee40       tb_ctrl with the replay Huber knee 40 (10 in tb_m4)
  tb_bwd25        tb_ctrl with fracs bwd 0.25, replay 0.75 (0.5, 0.5 in tb_m4)
  tb_fwd_alpha25  tb_ctrl with fresh rollouts training the policy by TB against the detached head (pool4_nov24
                  tb_m4_onpol's forward branch) at a share of 0.25 beside bwd and replay (0.375 each), every step, and
                  z_level_alpha 0.25. Compare with tb_alpha25
  cap_big         fresh MLE as mle_live with a larger flow (CAP_BIG: conditioner width, depth and heads, two spline
                  blocks, 64 bins, wider context), MLE_FRESH_STEPS: meant to overfit. Compare with mle_live and
                  pool4_nov26 cap_policy. Its smoke reports the batch it holds (`Batch Size`, `batch/oom_events`)
Chained (submit_pool4_nov27_chained.sbatch; each needs a checkpoint that does not exist yet, and its sbatch stops on a
missing source checkpoint):
  tb_T10_mleT10   TB at T 10 (tb_m4_T10's settings) from mle_T10's phase1_exit (MLE_T10_EXIT). Compare with tb_m4_T10
  tb_cap          tb_ctrl with pool4_nov26 cap_policy's flow, from its phase1_exit (CAP_POLICY_EXIT, written when
                  cap_policy finishes). Compare with tb_ctrl
smoke_*: `smoke: true`, one per arm: SMOKE_MLE_STEPS MLE steps, SMOKE_TB_STEPS TB steps (past the LR ramp).
Every MLE arm stops at its MLE stage's exit (snapshot:phase1_exit), every TB arm at its TB stage's (snapshot:tb_exit).
pool4_nov24 tb_m4_T10's own resume (configs/pool4_nov24/submit_pool4_nov24_hot_resume.sbatch, index 0) and pool4_nov26b
mt_ctrl (TB at 2.5 from tcond_t's phase1_exit) belong to this battery's reading but keep their own configs."""
import copy
import importlib.util
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, HERE.parent / rel / 'make.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


n26b = _load('n26b', 'pool4_nov26b')
n26, p4 = n26b.n26, n26b.p4

TAG = 'pool4_nov27'
MLE_WARM_STEPS, MLE_FRESH_STEPS, TB_STEPS = 80000, 150000, 40000
SMOKE_MLE_STEPS = 300
SMOKE_TB_STEPS = p4.RAMP_STEPS + 1000          # the TB LR ramp, then 1,000 steps at its full value
T_HOT = 10.0
CAP_BIG = dict(tf_width=1024, tf_layers=12, tf_heads=16, n_blocks=2, n_bins=64, context_dim=512,
               context_hidden=1024, hidden=1024)
#: checkpoint names as the trainer writes a snapshot, <tag>_<run_name>_<problem slug>_<snapshot>.pt, with the slug
#: utils.problem_slug of the writing yaml (computed 2026-11-27 from the generated yamls; T and the temperature-column
#: keys are part of the problem identity, the model is not). CHECK each file is in the cluster CKDIR before submitting
#: the chained arms.
MLE_T10_STEM = 'elj-pool8_4sg_train_k4_metric-T10-a59dfd'
MLE_T10_EXIT = f'{TAG}_mle_T10_{MLE_T10_STEM}_phase1_exit.pt'
CAP_POLICY_STEM = 'elj-pool8_4sg_train_k4_metric-T2.5-1d2dd7'
CAP_POLICY_EXIT = f'{n26.TAG}_cap_policy_{CAP_POLICY_STEM}_phase1_exit.pt'


def _hot(cfg, T=T_HOT):
    """energy_config.temperature T with the density wall scaled as pool4_nov24's hot arms; T is exempted on the
    weights-only load (warm_start_ignore_problem_keys), alongside any key the arm already exempts."""
    cfg['energy_config']['temperature'] = T
    cfg['energy_config']['density_coeff'] = p4.BASE_DENSITY * T / p4.BASE_T
    keep = [k for k in (cfg.get('warm_start_ignore_problem_keys') or []) if k != 'energy_config.temperature']
    cfg['warm_start_ignore_problem_keys'] = keep + ['energy_config.temperature']
    return cfg


def _tcond(cfg):
    """tcond_t's temperature conditioning: the column over log10 of n26b.T_RANGE, kicks drawn first over T_RANGE."""
    cfg['temperature_conditioning'] = True
    cfg['noised_prior']['kick_kappa_per_atom'] = n26.KAPPA_PER_ATOM
    cfg['noised_prior']['live_temperature_range'] = list(n26b.T_RANGE)
    cfg['energy_config']['log_temperature_range'] = list(n26b.LOG_T_RANGE)
    return cfg


def mle_T10(name, steps):
    cfg = _hot(n26.warm(name, steps))
    cfg['noised_prior']['kick_kappa_per_atom'] = n26.KAPPA_PER_ATOM
    cfg['noised_prior']['live_temperature_range'] = [T_HOT, T_HOT]
    assert cfg.get('temperature_conditioning') is not True
    cfg['tag'] = TAG
    return cfg


def tcond_t_cont(name, steps):
    cfg = n26b.tcond_t(name, steps)
    p4._set(cfg, 'checkpoint_name', n26b.TCOND_T_EXIT)
    assert cfg['load_weights_only'] is True
    cfg['tag'] = TAG
    return cfg


def tcond_fresh(name, steps):
    cfg = _tcond(n26.fresh(name, steps, {}))
    assert cfg.get('checkpoint_name') in (None, 'null')
    cfg['tag'] = TAG
    return cfg


def tb(name, steps, changes=()):
    """pool4_nov24 tb_m4 (mle4's final weights, MLE stage skipped) with the TB stage's LR ramped and `changes`."""
    cfg = p4.tb_arm(name, [('tb.lr_schedule', p4.ramp(0.2))] + list(changes), tb_steps=steps)
    cfg['fwd_loss_coeffs'].setdefault('z_level_alpha', 0.0)     # the base block carries the key a stage overrides
    cfg['tag'] = TAG
    return cfg


def tb_hot_from(name, steps, ckpt, tcond=False):
    """tb_m4_T10's settings (pool4_nov24 hot_changes(10), LR ramped) from `ckpt` instead of mle4."""
    cfg = p4.tb_arm(name, p4.hot_changes(T_HOT), tb_steps=steps)
    p4._set(cfg, 'checkpoint_name', ckpt)
    if tcond:
        _tcond(cfg)
    cfg = _hot(cfg)
    cfg['tag'] = TAG
    return cfg


def mt_T10(name, steps):
    cfg = tb_hot_from(name, steps, n26b.TCOND_T_EXIT, tcond=True)
    return cfg


def tb_T10_mleT10(name, steps):
    cfg = tb_hot_from(name, steps, MLE_T10_EXIT)
    cfg['noised_prior']['kick_kappa_per_atom'] = n26.KAPPA_PER_ATOM
    cfg['noised_prior']['live_temperature_range'] = [T_HOT, T_HOT]
    return cfg


#: the pathwise arms (PW_ARMS): rows per step per branch, the per-row parameter-gradient cap (multiples of the median),
#: the norm fractions
PW_ROWS, PW_ROW_CAP, PW_FRAC, PW_FRAC_HI = 32, 3.0, 0.1, 0.3
#: every pathwise key at its code default (gflownet_losses.py::PATHWISE_DEFAULTS), carried by each base block a stage
#: overrides
PW_OFF = dict(pathwise=0.0, pathwise_norm_frac=0.0, pathwise_entropy=1.0, pathwise_grad_clip=0.0, pathwise_rows=0,
              pathwise_row_cap=0.0)


def pw(name, steps, branches, frac):
    """tb_ctrl with the pathwise term on `branches` at norm fraction `frac` each (PW_ROWS rows, PW_ROW_CAP)."""
    on = dict(pathwise=0.0, pathwise_norm_frac=frac, pathwise_entropy=1.0, pathwise_grad_clip=0.0,
              pathwise_rows=PW_ROWS, pathwise_row_cap=PW_ROW_CAP)
    cfg = tb(name, steps, [(f'tb.loss_coeffs.{b}.{k}', v) for b in branches for k, v in on.items()])
    for b in ('fwd', 'bwd', 'replay'):
        block = cfg[f'{b}_loss_coeffs']
        for k, v in PW_OFF.items():
            block.setdefault(k, v)
    return cfg


#: tb_m4_onpol's forward branch (pool4_nov24 ONPOL) beside bwd and replay: TB on a fresh rollout every step against
#: the detached learned head, with emp_z fitting Z(c) on the same rollouts
FWD_POLICY = [('fwd_loss_coeffs.tb_detach_z', 0.0), ('tb.fracs', {'fwd': 0.25, 'bwd': 0.375, 'replay': 0.375}),
              ('tb.fwd_rollout_every', 0), ('tb.fwd_z_sidecar', False), ('tb.loss_coeffs.fwd.freeze_policy', 0.0),
              ('tb.loss_coeffs.fwd.tb', 1.0), ('tb.loss_coeffs.fwd.tb_detach_z', 1.0)]


def tb_cap(name, steps):
    cfg = tb(name, steps)
    p4._set(cfg, 'checkpoint_name', CAP_POLICY_EXIT)
    p4._set(cfg, 'model.nf.atom_tokens', 'split_pooled')        # cap_policy's, from mle_live (mle4's is 'split')
    cfg['model']['nf'].update(tf_width=768, tf_layers=12)
    return cfg


ARMS = {
    'mle_T10': lambda s: mle_T10('mle_T10', s),
    'tcond_t_cont': lambda s: tcond_t_cont('tcond_t_cont', s),
    'tcond_fresh': lambda s: tcond_fresh('tcond_fresh', s),
    'mt_T10': lambda s: mt_T10('mt_T10', s),
    'tb_ctrl': lambda s: tb('tb_ctrl', s),
    'tb_alpha25': lambda s: tb('tb_alpha25', s, [('tb.loss_coeffs.fwd.z_level_alpha', 0.25)]),
    'tb_alpha50': lambda s: tb('tb_alpha50', s, [('tb.loss_coeffs.fwd.z_level_alpha', 0.5)]),
    'tb_knee40': lambda s: tb('tb_knee40', s, [('tb.loss_coeffs.replay.beta', 40.0)]),
    'tb_bwd25': lambda s: tb('tb_bwd25', s, [('tb.fracs', {'fwd': 0.0, 'bwd': 0.25, 'replay': 0.75})]),
    'tb_fwd_alpha25': lambda s: tb('tb_fwd_alpha25', s, FWD_POLICY + [('tb.loss_coeffs.fwd.z_level_alpha', 0.25)]),
    'cap_big': lambda s: _tag(n26.fresh('cap_big', s, dict(CAP_BIG))),
}
PW_ARMS = {
    'pw_on': lambda s: pw('pw_on', s, ('fwd',), PW_FRAC),
    'pw_off': lambda s: pw('pw_off', s, ('bwd', 'replay'), PW_FRAC),
    'pw_both': lambda s: pw('pw_both', s, ('fwd', 'bwd', 'replay'), PW_FRAC),
    'pw_both_hi': lambda s: pw('pw_both_hi', s, ('fwd', 'bwd', 'replay'), PW_FRAC_HI),
}
CHAINED = {
    'tb_T10_mleT10': lambda s: tb_T10_mleT10('tb_T10_mleT10', s),
    'tb_cap': lambda s: tb_cap('tb_cap', s),
}
STEPS = {'mle_T10': MLE_WARM_STEPS, 'tcond_t_cont': MLE_WARM_STEPS, 'tcond_fresh': MLE_FRESH_STEPS,
         'cap_big': MLE_FRESH_STEPS}
MLE_ARMS = set(STEPS)


def _tag(cfg):
    cfg['tag'] = TAG
    return cfg


def _steps(name):
    return STEPS.get(name, TB_STEPS)


def _smoke(name):
    return SMOKE_MLE_STEPS if name in MLE_ARMS else SMOKE_TB_STEPS


def _configs(arms):
    out = {}
    for name, build in arms.items():               # smokes first: they take the first array indices
        sm = build(_smoke(name))
        sm.update(run_name=f'smoke_{name}', smoke=True, eval_period=100, archive_period=10 ** 6)
        out[sm['run_name']] = sm
    for name, build in arms.items():
        out[name] = build(_steps(name))
        out[name]['run_name'] = name
    return out


def configs():
    return _configs(ARMS), _configs(CHAINED), _configs(PW_ARMS)


def _write(cfgs, suffix=''):
    for name, cfg in cfgs.items():
        assert cfg['run_name'] == name and cfg['tag'] == TAG, (name, cfg['run_name'], cfg['tag'])
        (HERE / f'{name}.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8', newline='\n')
    text = p4.sbatch(list(cfgs)).replace(p4.TAG, TAG)
    (HERE / f'submit_{TAG}{suffix}.sbatch').write_text(text, encoding='utf-8', newline='\n')
    full = [n for n in cfgs if not n.startswith('smoke_')]
    for name in full:
        cfg = copy.deepcopy(cfgs[name])
        cfg.update(continue_from_checkpoint=True, checkpoint_name=None, load_weights_only=False)
        (HERE / f'{name}_resume.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8', newline='\n')
    (HERE / f'submit_{TAG}{suffix}_resume.sbatch').write_text(
        p4.sbatch([f'{n}_resume' for n in full]).replace(p4.TAG, TAG), encoding='utf-8', newline='\n')
    return list(cfgs)


if __name__ == '__main__':
    now, chained, pw_arms = configs()
    print('pw:', ' '.join(_write(pw_arms, '_pw')))
    print('now:', ' '.join(_write(now)))
    print('chained:', ' '.join(_write(chained, '_chained')))
