"""pool8_smpw_nov19: NF score matching and pathwise reverse KL on the pool8 P-1 eLJ TB battery (owner 2026-11-19, via the
hub: these two only). Every arm is pool8_tbrate_nov17's bigbuf (fwd_rollout_every 5, replay max_size 200k; the owner's
best rate arm) from pool8_big_nov15/big step 50,000, on the SWITCHED eLJ (the 2026-11-19 default): the prior is the
restamped file with stored forces (configs/pool8_prior_nov09/restamp_prior.sbatch, IN=pool8_p1_train_k4_metric.pt
FORCES=1, which writes <stem>_switched.pt), so the control is rerun on the same energy as the arms.

  control      bigbuf, switched energy
  sm<w>        + score matching on fwd, bwd and replay at weight w (score_rows 256 per branch, Huber knee 10; fwd and
               bwd forces live, replay forces recorded at admission)
  pw<w>c<k>    + pathwise reverse KL at weight w (sticking the landing, on the rollout's own draw), each row's gradient
               norm in x capped at k (pathwise_grad_clip)

Cut to four arms 2026-11-19 (owner, after the local 5k side-by-side on the K=0 setup, batch 120, scratch nov19/local5k):
score 1e-3 and 1e-2 reshaped the score without moving sample quality in 1k steps; score 1.0 moved it hard (P50 +11
nats at step 250, recovering after); pathwise 0.03 was inert at 1k; pathwise 0.1 unclipped aborted at step 173 on a
raw gradient norm of 1.3e6 (the lr_ctrl disaster bar), hence the clip on the strong pathwise arm.

Owner reading rule: for every arm report replay/tb_err against fwd/tb_err (replay below fwd = replay overfitting),
beside eval_test/logw_std_within, Excess Energy Nats P50 and eval_bwd/jensen_z - eval_fwd/jensen_z per 1k steps
against control over the same steps. Offline evidence and costs: the hub's notes folder,
nf_score_pathwise_design_nov19.md.

usage: python configs/pool8_smpw_nov19/make.py            (arm yamls + submit_pool8_smpw_nov19.sbatch, an array)"""
import copy
import importlib.util
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('tbrate', HERE.parent / 'pool8_tbrate_nov17' / 'make.py')
tbrate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tbrate)
big, big2 = tbrate.big, tbrate.big2

TAG = 'pool8_smpw_nov19'
PRIOR = big.PRIOR[:-len('.pt')] + '_switched.pt'
SM_WEIGHTS = (1e-2, 1.0)
SM_NO_ABS_BAR = 0.1              # score weights at or above this run without lr_control's absolute gradient bar
PW_ARMS = ((1.0, 100.0),)        # (weight, pathwise_grad_clip)
SCORE_ROWS = 256
SCORE_BETA = 10.0

#: the new keys' base values (= the code defaults: off); every arm carries them so a stage override names a base key
BASE = {'fwd_loss_coeffs': dict(score=0.0, score_beta=SCORE_BETA, score_rows=0, pathwise=0.0, pathwise_grad_clip=0.0),
        'bwd_loss_coeffs': dict(score=0.0, score_beta=SCORE_BETA, score_rows=0, score_force='live'),
        'replay_loss_coeffs': dict(score=0.0, score_beta=SCORE_BETA, score_rows=0, score_force='stored')}


def _tag(w):
    return f'{w:g}'.replace('.', 'p').replace('-', 'm')


def arms(files, source):
    bigbuf = tbrate.arms(files, source)['bigbuf']
    for block, keys in BASE.items():
        assert not set(keys) & set(bigbuf[block]), (block, set(keys) & set(bigbuf[block]))
        bigbuf[block].update(keys)
    out = {'control': bigbuf}
    for w in SM_WEIGHTS:
        cfg = copy.deepcopy(bigbuf)
        lc = tbrate._stage(cfg).setdefault('loss_coeffs', {})
        for mode in ('fwd', 'bwd', 'replay'):
            lc.setdefault(mode, {}).update(score=w, score_rows=SCORE_ROWS)
        if w >= SM_NO_ABS_BAR:
            # the score term's raw (pre-clip) gradient is a second derivative through the flow: at weight 1 its norm
            # sits above lr_control's absolute bar (1e6) from step 0 (cluster sm1, wandb 2vk1nplp: 4.48e6 on every
            # retry, aborted at step 33). The global clip bounds the step; the relative excursion bar still guards.
            assert cfg['lr_control']['hard_failure']['grad_abs'] == 1.0e6, cfg['lr_control']['hard_failure']
            cfg['lr_control']['hard_failure']['grad_abs'] = None
        out[f'sm{_tag(w)}'] = cfg
    for w, clip in PW_ARMS:
        cfg = copy.deepcopy(bigbuf)
        tbrate._stage(cfg).setdefault('loss_coeffs', {}).setdefault('fwd', {}).update(pathwise=w,
                                                                                       pathwise_grad_clip=clip)
        out[f'pw{_tag(w)}c{_tag(clip)}'] = cfg
    for name, cfg in out.items():
        # the source checkpoint names the unswitched prior; _switched is the same rows re-scored under the switched eLJ
        assert cfg['load_weights_only'] is True
        cfg.update(run_name=name, tag=TAG, warm_start_ignore_problem_keys=['prior_path'])
    return out


def sbatch(names):
    text = (HERE / 'submit_template.sbatch').read_text(encoding='utf-8')
    return (text.replace('@TAG@', TAG).replace('@ARMS@', ' '.join(names)).replace('@NARMS@', str(len(names) - 1))
            .replace('@DATA@', big.DATA).replace('@FILES@', ' '.join((PRIOR, big.CONDITIONS, big.TEST))))


if __name__ == '__main__':
    files = {'prior': f'{big.DATA}/{PRIOR}', 'conditions': f'{big.DATA}/{big.CONDITIONS}', 'test': f'{big.DATA}/{big.TEST}'}
    made = arms(files, big2.source_name(tbrate.SOURCE_STEP))
    for name, cfg in made.items():
        (HERE / f'{name}.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8', newline='\n')
    (HERE / f'submit_{TAG}.sbatch').write_text(sbatch(list(made)), encoding='utf-8', newline='\n')
    print(', '.join(made), f'and submit_{TAG}.sbatch in {HERE}')
