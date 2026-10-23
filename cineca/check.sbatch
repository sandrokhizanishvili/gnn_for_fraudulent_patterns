#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Step "check" (debug queue, ~5 min): versions, then every operator on a synthetic graph on the GPU -> "OK".
# Submitted by run_step.sh, which passes account, QoS, time, memory, cores and CODE_DIR.
set -eo pipefail
source "$CODE_DIR/cineca/_env.sh"
python "$CINECA_DIR/check_env.py"
