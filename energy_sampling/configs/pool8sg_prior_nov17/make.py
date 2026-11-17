"""pool8sg_prior_nov17: the K=4 noised training priors of the pool8sg_nov15 searches, one build per space group, on CPU.
The sharded recipe of configs/pool8_prior_nov09 ({calib,shard,merge}_pool8_prior.sbatch), with three changes:
  - the space group is set (build_noised_prior --set 'space_groups=[g]');
  - no calibration: every group uses the pool8 P-1 K=4 prior's log_hi (setup step reads it), so the groups merge;
  - the merge step refuses an output older than its inputs instead of silently keeping it, and shards are written
    through a temporary file.
Per group g, three jobs chained by submit_sg.sh: setup_sg<g> (lists + log_hi) -> shard_sg<g> (array of NSHARD CPU
tasks) -> merge_sg<g> (prior, held-out anchors, conditions, test file).

usage: python configs/pool8sg_prior_nov17/make.py      (renders setup_sg<g>.sbatch, shard_sg<g>.sbatch, merge_sg<g>.sbatch)"""
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
GROUPS = (14, 19, 61)
K = 4
NTASKS = 2899                 # search tasks per group (pool8sg_nov15: 2,899 chunks)
SETS = 29                     # training band sets per shard, as pool8_prior_nov09
NSHARD = math.ceil(NTASKS / SETS)
TSF = '0.36358'
DATA = '/scratch/mk8347/data/crystal_datasets/conditional'
P1_PRIOR = f'{DATA}/pool8_nov09/priors/pool8_p1_train_k{K}_metric.pt'


def fields(g):
    return {'@SG@': str(g), '@K@': str(K), '@SETS@': str(SETS), '@NSHARD@': str(NSHARD), '@LASTSHARD@': str(NSHARD - 1),
            '@NTASKS@': str(NTASKS), '@TSF@': TSF, '@P1_PRIOR@': P1_PRIOR,
            '@BAND@': f'{DATA}/pool8sg_nov15/band32/sg{g}', '@OUT@': f'{DATA}/pool8sg_nov15/priors/sg{g}',
            '@PRIOR@': f'pool8_sg{g}_train_k{K}_metric.pt', '@HELDOUT@': f'pool8_sg{g}_heldout_anchors.pt',
            '@CONDITIONS@': f'pool8_sg{g}_train_conditions.pt', '@TEST@': f'pool8_sg{g}_test_w5.pt',
            '@TEST_STEM@': f'pool8_sg{g}_test'}


def main():
    for g in GROUPS:
        for step in ('setup', 'shard', 'merge'):
            text = (HERE / f'{step}_template.sbatch').read_text(encoding='utf-8')
            for k, v in fields(g).items():
                text = text.replace(k, v)
            assert not re.search(r'@[A-Z_0-9]+@', text), f'{step} sg{g}: unfilled placeholder'
            (HERE / f'{step}_sg{g}.sbatch').write_text(text, encoding='utf-8', newline='\n')
    print(f'rendered setup/shard/merge for groups {GROUPS}: K {K}, {NSHARD} shards of {SETS} band sets')


if __name__ == '__main__':
    main()
