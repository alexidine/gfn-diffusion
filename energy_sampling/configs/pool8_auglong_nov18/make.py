"""pool8_auglong_nov18: the long augmented MLE run (owner 2026-11-18). A short unaugmented warm-up, then augmented MLE
until held-out stops improving, then a learning-rate decay. Stage 1 of that recipe is pool8_big2_nov17's aug10k arm
(10k unaugmented steps of pool8_big_nov15/big, then 15k augmented). This run continues it, weights only, from its
phase1_exit snapshot. One arm, 'auglong': aug10k's config (pool8_big2_nov17/make.py: aug_config) with:

  train_prior   augmented MLE at the held LR (0.2 x seed), exit at the first eval where gates/heldout_flat holds, on
                heldout_gate source eval_bwd/mle_heldout_rows_frame_mean (held-out crystals, random-frame -log q,
                train.py::frame_readouts), evals 10, nats 0.15, smooth 3: no 0.15-nat gain in 5,000 steps.
                Bound: max_steps PLATEAU_CAP.
  anneal        the same losses, LR geometric from 0.2 to 0.02 x seed over ANNEAL_STEPS, then snapshot and stop.

Step labels continue aug10k's own axis from 15,000 (aug10k predates step_offset, so its axis starts at its own 0);
the total since the unaugmented start is the label + 10,000. Caps from aug10k's measured ~0.93 steps/s: 125k steps
is ~37 h of the 2-day limit.

usage: python configs/pool8_auglong_nov18/make.py              (auglong.yaml + submit_pool8_auglong_nov18.sbatch)
       python configs/pool8_auglong_nov18/make.py --local DIR --source CKPT --ckdir DIR [--steps N]"""
import argparse
import copy
import importlib.util
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('big2', HERE.parent / 'pool8_big2_nov17' / 'make.py')
big2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(big2)
big = big2.big
nov10 = big2.nov10

TAG = 'pool8_auglong_nov18'
SOURCE = f'pool8_big2_nov17_aug10k_{big2.SOURCE_SLUG}_phase1_exit.pt'
PLATEAU_CAP = 115000
ANNEAL_STEPS = 10000
HELDOUT_GATE = {'source': 'eval_bwd/mle_heldout_rows_frame_mean', 'evals': 10, 'nats': 0.15, 'smooth': 3}


def auglong(base, source):
    cfg = big2.aug_config(base, source)
    stages = cfg['protocols'][cfg['protocol']]['stages']
    assert [s['name'] for s in stages] == ['train_prior'], [s['name'] for s in stages]
    tp = stages[0]
    assert tp['lr_schedule'] == [{'to': 0.2, 'steps': 0, 'kind': 'hold'}] and tp['on_exit'][-1] == 'stop', tp
    anneal = copy.deepcopy(tp)
    tp.update(max_steps=PLATEAU_CAP, on_exit=['snapshot:plateau_exit'],
              exit=[{'metric': 'gates/heldout_flat', 'above': 0.5, 'patience': 1}])
    anneal.pop('exit', None)
    anneal.update(name='anneal', max_steps=ANNEAL_STEPS, on_exit=['snapshot:anneal_exit', 'stop'],
                  lr_schedule=[{'to': 0.2, 'steps': 0, 'kind': 'hold'},
                               {'to': 0.02, 'steps': ANNEAL_STEPS, 'kind': 'geometric'}])
    stages.append(anneal)
    cfg.update(heldout_gate=dict(HELDOUT_GATE), epochs=PLATEAU_CAP + ANNEAL_STEPS, archive_period=2500,
               run_name='auglong', tag=TAG)
    return cfg


def sbatch(names):
    text = (HERE / 'submit_template.sbatch').read_text(encoding='utf-8')
    files = (big.PRIOR, big.CONDITIONS, big.TEST)
    return (text.replace('@TAG@', TAG).replace('@ARMS@', ' '.join(names)).replace('@NARMS@', str(len(names) - 1))
            .replace('@DATA@', big.DATA).replace('@FILES@', ' '.join(files)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--local', default=None, help='write a local smoke config (norbornane files) into this directory')
    p.add_argument('--source', default=None, help='local smoke: checkpoint file name (in --ckdir) to load')
    p.add_argument('--ckdir', default=None, help='local smoke: checkpoints directory (reads the source, writes there)')
    p.add_argument('--steps', type=int, default=60)
    a = p.parse_args()
    if a.local:
        assert a.source and a.ckdir, '--local needs --source and --ckdir'
        out = Path(a.local)
        out.mkdir(parents=True, exist_ok=True)
        cfg = auglong(big.config(nov10.LOCAL), a.source)
        # smoke: a tiny gate (2 evals, never flat) and short stages so both stages and the transition run
        st = cfg['protocols'][cfg['protocol']]['stages']
        st[0]['max_steps'], st[1]['max_steps'] = a.steps // 2, a.steps // 2
        st[1]['lr_schedule'][1]['steps'] = a.steps // 2
        cfg.update(checkpoints_dir=a.ckdir, tag=f'{TAG}_local', smoke=True, cuda_memory_fraction=0.5, epochs=a.steps + 20,
                   batch_size=min(cfg['batch_size'], 64), eval_period=max(10, a.steps // 6), archive_period=10 ** 6)
        (out / 'auglong.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8', newline='\n')
        print(f'local smoke config in {out}')
    else:
        files = {'prior': f'{big.DATA}/{big.PRIOR}', 'conditions': f'{big.DATA}/{big.CONDITIONS}',
                 'test': f'{big.DATA}/{big.TEST}'}
        cfg = auglong(big.config(files), SOURCE)
        (HERE / 'auglong.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8', newline='\n')
        (HERE / f'submit_{TAG}.sbatch').write_text(sbatch(['auglong']), encoding='utf-8', newline='\n')
        print(f'auglong.yaml and submit_{TAG}.sbatch in {HERE}')
