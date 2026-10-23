#!/bin/bash
# Usage (on Sandro's laptop, in WSL Ubuntu): bash cineca/make_env_pack.sh [output folder]
# Backup for the case pip fails on Cineca: builds a conda environment with the same requirements.lock (Python 3.12
# from conda, every package from pip), packs it with conda-pack into handoff_env.tar.gz and prints its size.
# On Cineca: mkdir -p $ENV_DIR && tar -xzf handoff_env.tar.gz -C $ENV_DIR && source $ENV_DIR/bin/activate && conda-unpack
# (README_LORIS.md section 2). Not shipped to Cineca.
set -eo pipefail
CINECA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-$CINECA_DIR/../../handoff_cineca}"
command -v conda > /dev/null || { echo "conda not found in WSL: install Miniconda first (https://docs.conda.io/en/latest/miniconda.html)"; exit 1; }
PREFIX="$HOME/aml_thesis_env"
if [ ! -d "$PREFIX" ]; then
    conda create -y -p "$PREFIX" python=3.12
fi
"$PREFIX/bin/python" -m pip install --no-cache-dir --prefer-binary \
    --only-binary=torch,torch-scatter,torch-sparse,torch-cluster -r "$CINECA_DIR/requirements.lock"
"$PREFIX/bin/python" -m pip freeze > "$PREFIX/pip_freeze.txt"
if ! conda pack --help > /dev/null 2>&1; then
    conda install -y -n base -c conda-forge conda-pack
fi
mkdir -p "$OUT"
conda pack -p "$PREFIX" -o "$OUT/handoff_env.tar.gz" --force
echo "handoff_env.tar.gz: $(du -h "$OUT/handoff_env.tar.gz" | cut -f1) -> $OUT"
