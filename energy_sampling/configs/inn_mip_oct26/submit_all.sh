#!/bin/bash
# Submits inn_mip_oct26's six jobs, one A100 each (arms_p1 .. arms_a3).
cd "$(dirname "$0")"
for g in p1 p2 p3 a1 a2 a3; do sbatch --export=ALL,ARMS_TSV=arms_${g}.tsv submit_inn_mip26.sbatch; done
