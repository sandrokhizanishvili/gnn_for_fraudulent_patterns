#!/bin/bash
# Usage: bash cineca/status.sh — one table: done / running / queued / timeout / failed / missing per step and
# (operator, config).
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
python "$CINECA_DIR/status.py"
