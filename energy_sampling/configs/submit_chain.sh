#!/bin/bash
# Submit one sbatch N times as a dependency chain, so a battery keeps running
# across the 48 h wall limit with nobody at a terminal.
#
#     configs/submit_chain.sh configs/acr_c_aug31/submit_acr_c_aug31.sbatch 3
#
# Each leg resumes the previous leg's own _running.pt, because every battery
# sbatch already globs `*${ARM}_*_running.pt` before falling back to the shared
# phase-1 exit. So the chain needs no cooperation from the sbatch itself and
# works with any battery in this tree, including ones already written.
#
# WHY afterany AND NOT aftercorr.
# `aftercorr` is the tempting one: it pairs array task k of leg N+1 to array
# task k of leg N, so fast arms would advance without waiting for slow ones.
# It is WRONG here. SLURM's aftercorr fires only when the corresponding task
# "completed successfully (ran to completion with an exit code of zero)", and a
# job killed at its wall limit does not exit zero -- CANCELLED ... DUE TO TIME
# LIMIT is precisely the case the chain exists to survive. An aftercorr chain
# would therefore stall on exactly the event it was built for, silently, with
# the remaining legs sitting in DependencyNeverSatisfied.
#
# `afterany` fires on ANY terminal state, which is what we want, at the cost of
# being whole-array: leg N+1 starts when the last task of leg N ends. Within a
# battery whose arms have similar step times that costs little, and correctness
# beats packing here -- an unattended chain that stops is worth less than one
# that runs slightly ragged.
#
# COST OF A LEG THAT DIES EARLY: afterany also fires on a crash, so a config
# error would burn all N legs in quick succession rather than one. That is the
# accepted trade -- the alternative stalls on a timeout. Check the first leg
# started cleanly before walking away.
set -euo pipefail

SB="${1:?usage: submit_chain.sh <sbatch-file> [n_legs]}"
N="${2:-3}"

[ -f "$SB" ] || { echo "no such sbatch: $SB" >&2; exit 1; }
case "$N" in ''|*[!0-9]*) echo "n_legs must be an integer, got: $N" >&2; exit 1;; esac
[ "$N" -ge 1 ] || { echo "n_legs must be >= 1" >&2; exit 1; }

prev=""
for i in $(seq 1 "$N"); do
    if [ -z "$prev" ]; then
        id=$(sbatch --parsable "$SB")
        echo "leg $i/$N: job $id"
    else
        id=$(sbatch --parsable --dependency=afterany:"$prev" "$SB")
        echo "leg $i/$N: job $id   (starts after $prev, any exit state)"
    fi
    prev="$id"
done

echo
echo "chain submitted. cancel the whole thing with:  scancel --name=\$(awk '/--job-name/{sub(/.*=/,\"\");print}' $SB)"
echo "NB every leg resumes its arm's own _running.pt; the last leg to run wins."
