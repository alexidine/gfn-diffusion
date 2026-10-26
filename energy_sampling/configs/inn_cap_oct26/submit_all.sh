#!/bin/bash
# Submits inn_cap_oct26's four jobs, one A100 each (arms_s, arms_m, arms_l1, arms_l2).
cd "$(dirname "$0")"
for g in s m l1 l2; do sbatch --export=ALL,ARMS_TSV=arms_${g}.tsv submit_inn_cap26.sbatch; done
