# Sourced by run.sh and job.sbatch: settings (config.sh, then config.local.sh), modules, the Python environment.
CINECA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(dirname "$CINECA_DIR")"
source "$CINECA_DIR/config.sh"
if [ -f "$CINECA_DIR/config.local.sh" ]; then
    source "$CINECA_DIR/config.local.sh"
fi
if [ -n "${MODULES:-}" ]; then
    module load $MODULES
fi
if [ ! -x "$ENV_DIR/bin/python" ]; then
    echo "No Python environment at ENV_DIR=$ENV_DIR: create it first (README section 2)" >&2
    exit 1
fi
export PATH="$ENV_DIR/bin:$PATH"
export MPLBACKEND=Agg              # plots go to files, never to a screen
export PYTHONUNBUFFERED=1          # log lines appear as they happen
export TQDM_DISABLE=1              # no progress bars in the logs
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-$CPUS}"
export CODE_DIR CINECA_DIR
