# pool8sg_prior_nov17

K=4 noised training priors for the pool8sg_nov15 searches in P2_1/c (14), P2_12_12_1 (19) and Pbca (61): one build per
group, CPU only. Owner 2026-11-17: keep K=4 (the target the P-1 big run learned), build on CPU, groups 14, 19, 61.

The recipe is configs/pool8_prior_nov09's sharded build (calib, shard array, merge), which had never run on the cluster:
the P-1 K=4 prior came from its GPU build. Changes: the space group is set (`--set 'space_groups=[g]'`); no calibration
step (every group takes the P-1 K=4 prior's `noised_build.log_hi`, so the groups' files can be merged); shards are
written through a temporary file; the merge step stops on an output older than its inputs instead of keeping it.

Per group g (make.py renders them; edit the templates, not the rendered files):

| step | job | resources | what it does |
|---|---|---|---|
| 1 | setup_sg<g>.sbatch | 1 core, 16 GB, 1 h | checks all 2,899 band records exist; writes train/held-out band-set lists; reads log_hi from the P-1 prior (memory-mapped) |
| 2 | shard_sg<g>.sbatch | array 0-99, 4 cores, 16 GB, 6 h | shard i = band sets 29i+1 .. 29i+29: K 4 metric children, seed 1000 + i |
| 3 | merge_sg<g>.sbatch | 8 cores, 96 GB, 8 h | merged prior, held-out anchors (K 0), conditions file, test file (W 5, 8 per molecule, 500 molecules) |

Outputs in `/scratch/mk8347/data/crystal_datasets/conditional/pool8sg_nov15/priors/sg<g>/`:
`pool8_sg<g>_train_k4_metric.pt`, `pool8_sg<g>_heldout_anchors.pt`, `pool8_sg<g>_train_conditions.pt`,
`pool8_sg<g>_test_w5.pt`.

Launch (on the cluster, once a group's searches have banded every task):

    cd /scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/pool8sg_prior_nov17
    bash submit_sg.sh 14 19 61

Checked locally 2026-11-17 on the P2_1/c smoke band sets (3 molecules): shard build, two-shard merge, held-out anchors,
conditions and test files ran with these commands; a trainer smoke (the pool8_big_nov15 model, space_groups [14], 60
MLE steps) loaded the files and trained. The memory-mapped log_hi read was checked on a local prior file. Not run on the
cluster.
