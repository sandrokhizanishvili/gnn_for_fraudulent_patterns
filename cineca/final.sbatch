#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Step "final": one array per (OPERATOR, CONFIG); array index i -> seed 42 + i; see final.py.
# Submitted by run_step.sh, which passes account, QoS, time, memory, cores, the array, OPERATOR, CONFIG and CODE_DIR.
set -eo pipefail
source "$CODE_DIR/cineca/_env.sh"
python "$CINECA_DIR/final.py" --operator "$OPERATOR" --config "$CONFIG" --index "$SLURM_ARRAY_TASK_ID"
