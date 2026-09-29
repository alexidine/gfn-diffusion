"""capped_mc_sep29: capped Metropolis walkers (data_processing/capped_mc.py) from each MLIP prior's search minima, on
the training energy, for the thermal-band prior rebuild. Writes INDEX.tsv (one row per (system, shard) array task) and
rewrites the --array line of submit_capped_mc.sbatch.

    python configs/capped_mc_sep29/make.py

Each task thins its system's windowed minima at the same-packing RDF radius (identically in every shard), takes a
round-robin share of the distinct minima and runs `replicas` walkers per minimum; every task runs --resume, so
resubmitting the same file continues each shard from its state.pt. Exploration is logged per checkpoint in each shard's
exploration.jsonl (report: python data_processing/capped_mc_explore.py <run dir>).
PILOT=1 (env at sbatch) runs a small version (64 minima, 20 steps) into <name>_pilot/: a smoke test of the setup.
The acridine row seeds from the CURRENT MACE prior (the old compressed conformer); for the redo it should point at the
new search export once that exists.
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
UMA = '/scratch/mk8347/models/uma/esen_s.pt'
MACE = '/scratch/mk8347/data/acr_112025_mh1_stagetwo.model'
# name, prior file (under /scratch/mk8347/data/crystal_datasets/conditional/priors), its size in bytes (the copy the
# p23 runs trained on; the sbatch refuses a different file), energy function, MLIP checkpoint, shards, RDF mode,
# same-packing RDF radius (COMPACK-calibrated: minima are thinned at it; exploration is tracked at 1, 2, 3 x it),
# walkers per distinct minimum
SYSTEMS = [
    ('mipu_uma', 'mipcas_sg2_zp1_uma_f047_200k_prior_dataset_niggli_v2.pt', 232846061, 'uma', UMA, 2, 'atomwise', 0.116, 2),
    ('nehu_uma', 'nehzor_sg14_zp1_uma_f047_200k_prior_dataset_w3.pt', 252737846, 'uma', UMA, 2, 'atomwise', 0.133, 2),
    ('acr_mace', 'acridine_sg14_zp1_mace_prior_dataset_w3.pt', 270223697, 'mace', MACE, 4, 'envwise', 0.085, 2),
]
COLS = ['task', 'name', 'shard', 'n_shards', 'prior', 'prior_bytes', 'energy_function', 'mlip_path', 'rdf_mode',
        'rdf_dcut', 'replicas']
DRIVE = re.compile(r'(^|[\s\'"=])[A-Za-z]:[\\/]')


def assert_no_local_paths(text, where):
    """a drive-letter path passes every local check (the file exists on the dev box) and kills the job on-cluster"""
    bad = [ln for ln in text.splitlines() if DRIVE.search(ln)]
    if bad:
        raise SystemExit(f'{where}: local path(s) would ship to the cluster:\n  ' + '\n  '.join(bad))


rows = ['\t'.join(COLS)]
t = 0
for name, prior, nbytes, ef, mlip, n, mode, dcut, reps in SYSTEMS:
    for k in range(n):
        rows.append('\t'.join(str(v) for v in (t, name, k, n, prior, nbytes, ef, mlip, mode, dcut, reps)))
        t += 1
index = '\n'.join(rows) + '\n'
assert_no_local_paths(index, 'INDEX.tsv')
open(os.path.join(HERE, 'INDEX.tsv'), 'w', newline='\n').write(index)

sb = os.path.join(HERE, 'submit_capped_mc.sbatch')
text = open(sb, encoding='utf8').read()
text = re.sub(r'#SBATCH --array=\S+', f'#SBATCH --array=0-{t - 1}', text)
assert_no_local_paths(text, 'submit_capped_mc.sbatch')
open(sb, 'w', encoding='utf8', newline='\n').write(text)
print(f'{t} tasks: ' + ', '.join(f'{s[0]} x{s[5]}' for s in SYSTEMS))
