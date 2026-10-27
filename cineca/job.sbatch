#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
# The one job script: runs "luo.py $STEP" with this task's array index. Submitted by run.sh, which passes account,
# QoS, time, memory, cores, the array and STEP (+ OPERATOR, CONFIG for tune and final) through --export.
set -eo pipefail
source "$CODE_DIR/cineca/env.sh"
python "$CINECA_DIR/luo.py" "$STEP" ${SLURM_ARRAY_TASK_ID:+--index "$SLURM_ARRAY_TASK_ID"} \
    ${OPERATOR:+--operator "$OPERATOR"} ${CONFIG:+--config "$CONFIG"}
