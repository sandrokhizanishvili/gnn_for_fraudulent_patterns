#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Step "probe": array index i -> operator (0 gin, 1 pna, 2 gat, 3 transformer); see probe.py.
# Submitted by run_step.sh, which passes account, QoS, time, memory, cores, the array and CODE_DIR.
set -eo pipefail
source "$CODE_DIR/cineca/_env.sh"
python "$CINECA_DIR/probe.py" --index "$SLURM_ARRAY_TASK_ID"
