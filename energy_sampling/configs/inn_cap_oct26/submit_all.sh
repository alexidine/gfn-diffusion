#!/bin/bash
# Submits inn_cap_oct26's three jobs, one A100 each (arms_s, arms_m, arms_l).
cd "$(dirname "$0")"
for g in s m l; do sbatch --export=ALL,ARMS_TSV=arms_${g}.tsv submit_inn_cap26.sbatch; done
