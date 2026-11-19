"""pool8_smpw_nov19: NF score matching and pathwise reverse KL on the pool8 P-1 eLJ TB battery (owner 2026-11-19, via the
hub: these two only). Every arm is pool8_tbrate_nov17's bigbuf (fwd_rollout_every 5, replay max_size 200k; the owner's
best rate arm) from pool8_big_nov15/big step 50,000, on the SWITCHED eLJ (the 2026-11-19 default): the prior is the
restamped file with stored forces (configs/pool8_prior_nov09/restamp_prior.sbatch, IN=pool8_p1_train_k4_metric.pt
FORCES=1, which writes <stem>_switched.pt), so the control is rerun on the same energy as the arms.

  control      bigbuf, switched energy
  sm<w>        + score matching on fwd, bwd and replay at weight w (score_rows 256 per branch, Huber knee 10; fwd and
               bwd forces live, replay forces recorded at admission)
  pw<w>        + pathwise reverse KL at weight w (sticking the landing, on the rollout's own draw)

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
SM_WEIGHTS = (1e-4, 1e-3, 1e-2)
PW_WEIGHTS = (0.01, 0.03, 0.1)
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
        out[f'sm{_tag(w)}'] = cfg
    for w in PW_WEIGHTS:
        cfg = copy.deepcopy(bigbuf)
        tbrate._stage(cfg).setdefault('loss_coeffs', {}).setdefault('fwd', {})['pathwise'] = w
        out[f'pw{_tag(w)}'] = cfg
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
