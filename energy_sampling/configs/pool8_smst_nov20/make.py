"""pool8_smst_nov20: strong NF score matching on the pool8 P-1 eLJ TB battery (owner 2026-11-20: "we didn't get a good
picture yet of strong score matching"). pool8_smpw_nov19's control recipe (pool8_tbrate_nov17 bigbuf from
pool8_big_nov15/big step 50,000 on the switched prior <stem>_switched.pt, 10k steps, rollouts every 5), with the two
confounds of that battery removed from EVERY arm:
  * batch pinned at 551 (grow off): the OOM cutter left nov19's arms at different batches (1412 / 882 / 551 / 344);
    551 is the size every arm reached there
  * lr_control.hard_failure.grad_abs null: the score term's raw gradient is a second derivative through the flow and
    sits above the 1e6 absolute bar (nov19 sm1 4.48e6, sm0p01 1.93e6 at step 8,063); the global clip and the relative
    excursion bar still guard

  control      the recipe
  sm<w>        + score matching on fwd, bwd and replay at weight w (score_rows 256 per branch, Huber knee 10; fwd and
               bwd forces live, replay forces recorded at admission)
  sm1bwd       + score matching at 1.0 on the stored (bwd) rows only: locally the stored-row term converged while the
               fresh and replay rows saturated against stiff forces; this separates the two

Read per 1k steps against control: eval_fwd/jensen_z, Mean Sample Energy, Excess Energy Nats P50, Reasonable Sample
Fraction, eval_test/logw_std_within, replay/tb_err vs fwd/tb_err, and the arms' <branch>/score_gq_norm /
score_gr_norm and score_cos.

usage: python configs/pool8_smst_nov20/make.py            (arm yamls + submit_pool8_smst_nov20.sbatch, an array)"""
import copy
import importlib.util
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('smpw', HERE.parent / 'pool8_smpw_nov19' / 'make.py')
smpw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(smpw)
tbrate, big, big2 = smpw.tbrate, smpw.big, smpw.big2

TAG = 'pool8_smst_nov20'
BATCH = 551
SM_WEIGHTS = (0.1, 0.3, 1.0)
SCORE_ROWS = 256


def arms(files, source):
    control = smpw.arms(files, source)['control']
    assert control['grow_batch_size'] is False and control['lr_control']['hard_failure']['grad_abs'] == 1.0e6
    control['batch_size'] = BATCH
    control['lr_control']['hard_failure']['grad_abs'] = None
    out = {'control': control}
    for w in SM_WEIGHTS:
        cfg = copy.deepcopy(control)
        lc = tbrate._stage(cfg).setdefault('loss_coeffs', {})
        for mode in ('fwd', 'bwd', 'replay'):
            lc.setdefault(mode, {}).update(score=w, score_rows=SCORE_ROWS)
        out[f'sm{smpw._tag(w)}'] = cfg
    cfg = copy.deepcopy(control)
    tbrate._stage(cfg).setdefault('loss_coeffs', {}).setdefault('bwd', {}).update(score=1.0, score_rows=SCORE_ROWS)
    out['sm1bwd'] = cfg
    for name, cfg in out.items():
        cfg.update(run_name=name, tag=TAG)
    return out


def sbatch(names):
    text = (HERE / 'submit_template.sbatch').read_text(encoding='utf-8')
    return (text.replace('@TAG@', TAG).replace('@ARMS@', ' '.join(names)).replace('@NARMS@', str(len(names) - 1))
            .replace('@DATA@', big.DATA).replace('@FILES@', ' '.join((smpw.PRIOR, big.CONDITIONS, big.TEST))))


if __name__ == '__main__':
    files = {'prior': f'{big.DATA}/{smpw.PRIOR}', 'conditions': f'{big.DATA}/{big.CONDITIONS}',
             'test': f'{big.DATA}/{big.TEST}'}
    made = arms(files, big2.source_name(tbrate.SOURCE_STEP))
    for name, cfg in made.items():
        (HERE / f'{name}.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8', newline='\n')
    (HERE / f'submit_{TAG}.sbatch').write_text(sbatch(list(made)), encoding='utf-8', newline='\n')
    print(', '.join(made), f'and submit_{TAG}.sbatch in {HERE}')
