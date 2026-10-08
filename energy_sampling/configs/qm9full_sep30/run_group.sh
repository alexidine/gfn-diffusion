#!/bin/bash
# Run the arms of one job on its one GPU, each as its own trainer (qm9full_sep30 leg j; written by make.py).
# usage, inside the container and from energy_sampling/:
#     run_group.sh <logs dir> <job id> <share file> <arm> [<arm> ...]
# An arm's config is <logs dir>/<arm>_<job id>.yaml and its output <logs dir>/<arm>_<job id>.trainlog.
# One arm runs as it would alone. More than one share the card through NVIDIA MPS when the daemon comes up
# and a probe process reaches the card through it; otherwise by plain time slicing. The first line of
# <share file> says which.
LOGS=$1; JOB=$2; SHARE=$3; shift 3
ARMS=("$@")
N=${#ARMS[@]}
if [ ${N} -lt 1 ]; then echo "run_group.sh: no arm given" >&2; exit 1; fi

probe() {
    python -c "import torch; torch.zeros(8, device='cuda').sum().item(); print('probe: cuda ok')"
}

MODE=solo
if [ ${N} -gt 1 ]; then
    MODE=timeslice
    if command -v nvidia-cuda-mps-control > /dev/null 2>&1; then
        export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_${JOB}/pipe
        export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_${JOB}/log
        mkdir -p ${CUDA_MPS_PIPE_DIRECTORY} ${CUDA_MPS_LOG_DIRECTORY}
        nvidia-cuda-mps-control -d
        # the daemon is up when its control pipe exists: ten seconds, then the arms run without it
        for i in $(seq 1 20); do
            [ -e ${CUDA_MPS_PIPE_DIRECTORY}/control ] && break
            sleep 0.5
        done
        if [ -e ${CUDA_MPS_PIPE_DIRECTORY}/control ] && probe; then
            MODE=mps
        else
            echo "MPS did not come up or the probe could not reach the card through it: time slicing instead"
            echo quit | nvidia-cuda-mps-control 2> /dev/null
            unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
        fi
    else
        echo "nvidia-cuda-mps-control is not in this container: time slicing instead"
    fi
fi
echo "GPU sharing for ${N} arm(s): ${MODE}" | tee ${SHARE}
if [ "${MODE}" = timeslice ] && ! probe; then
    echo "FATAL: the card cannot be reached without MPS either" | tee -a ${SHARE} >&2; exit 1
fi

# A signal from the scheduler reaches every process of the job: the trainers handle their own, and this shell
# must outlive them or the container closes under their last checkpoint write.
trap ':' TERM INT

PIDS=()
for ARM in "${ARMS[@]}"; do
    python -u train.py --config ${LOGS}/${ARM}_${JOB}.yaml > ${LOGS}/${ARM}_${JOB}.trainlog 2>&1 &
    PIDS+=($!)
    echo "started ${ARM} (pid ${PIDS[-1]})"
done
# what holds the card once every trainer is up: under MPS the server is listed beside its clients
# (slept in short pieces: killed below when the trainers end first, it leaves nothing waiting behind it)
( for i in $(seq 1 60); do sleep 5; done
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >> ${SHARE} 2>&1 ) &
NOTE_PID=$!

STATUS=0
for i in "${!PIDS[@]}"; do
    # `wait` returns early, above 128, when a trapped signal arrives: go round until the trainer has gone.
    # It ends when the trainer does, and the scheduler's kill after its grace period bounds that.
    while :; do
        wait ${PIDS[$i]}; RC=$?
        kill -0 ${PIDS[$i]} 2> /dev/null || break
    done
    echo "arm ${ARMS[$i]} ended with status ${RC}" | tee -a ${SHARE}
    [ ${RC} -ne 0 ] && STATUS=${RC}
done
kill ${NOTE_PID} 2> /dev/null
[ "${MODE}" = mps ] && echo quit | nvidia-cuda-mps-control
exit ${STATUS}
