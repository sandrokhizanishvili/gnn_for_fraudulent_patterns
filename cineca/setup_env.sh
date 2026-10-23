#!/bin/bash
# Usage: bash cineca/setup_env.sh   (once, on a LOGIN node: it downloads packages, compute nodes have no internet)
# Creates the Python environment in ENV_DIR from requirements.lock, records the installed versions, and runs the
# CPU part of the environment check. The GPU part is "bash cineca/run_step.sh check" afterwards.
# If pip gets killed (login-node CPU-time limit), run this script again: packages already installed are skipped.
set -eo pipefail
CINECA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$CINECA_DIR/config.sh"
if [ -n "${MODULES:-}" ]; then
    module load $MODULES
fi
echo "using $(command -v python3) ($(python3 --version))"
if [ ! -f "$ENV_DIR/bin/activate" ]; then
    python3 -m venv "$ENV_DIR"
fi
source "$ENV_DIR/bin/activate"
mkdir -p "$BASE/tmp"
export TMPDIR="$BASE/tmp"          # torch wheels are large; the default /tmp may be too small
python -m pip install --no-cache-dir --upgrade pip
python -m pip install --no-cache-dir --prefer-binary --only-binary=torch,torch-scatter,torch-sparse,torch-cluster \
    -r "$CINECA_DIR/requirements.lock"
python -m pip freeze > "$ENV_DIR/pip_freeze.txt"
echo "installed packages -> $ENV_DIR/pip_freeze.txt"
MPLBACKEND=Agg REQUIRE_GPU=0 python "$CINECA_DIR/check_env.py"
echo "environment ready in $ENV_DIR. Next: bash cineca/run_step.sh check"
