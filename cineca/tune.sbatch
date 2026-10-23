#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Step "tune": one array per (OPERATOR, CONFIG); array index i -> trial i // 2, seed 42 + i % 2; see tune.py.
# Submitted by run_step.sh, which passes account, QoS, time, memory, cores, the array, OPERATOR, CONFIG and CODE_DIR.
set -eo pipefail
source "$CODE_DIR/cineca/_env.sh"
python "$CINECA_DIR/tune.py" --operator "$OPERATOR" --config "$CONFIG" --index "$SLURM_ARRAY_TASK_ID"
