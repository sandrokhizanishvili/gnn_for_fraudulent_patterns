# Sourced by every job script and by run_step.sh / status.sh / pack_results.sh: settings, modules, Python env.
# Nothing here needs editing — settings live in config.sh.
CINECA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(dirname "$CINECA_DIR")"
source "$CINECA_DIR/config.sh"
if [ -n "${MODULES:-}" ]; then
    module load $MODULES
fi
if [ ! -f "$ENV_DIR/bin/activate" ]; then
    echo "No Python environment at ENV_DIR=$ENV_DIR: run 'bash cineca/setup_env.sh' first (README section 2)" >&2
    exit 1
fi
source "$ENV_DIR/bin/activate"
export MPLBACKEND=Agg              # plots go to files, never to a screen
export PYTHONUNBUFFERED=1          # log lines appear as they happen
export TQDM_DISABLE=1              # no progress bars in the logs
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-$CPUS}"
export CODE_DIR CINECA_DIR
