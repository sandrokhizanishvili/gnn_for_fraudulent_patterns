#!/bin/bash
# Usage: bash cineca/pack_results.sh
# Runs the aggregate step (correctness checks + summaries; a failed check is reported, the packing still happens),
# then writes three archives next to OUT_DIR and prints their sizes:
#   results_small.tar.gz        every small result file + the job logs   (send this one first; the early check-in)
#   results_predictions.tar.gz  every predictions.csv of the final retrain
#   results_models.tar          every kept checkpoint: best.pt per seed of each trial, final.pt per final seed
#                               (not gzipped: model weights barely compress)
# Archive paths start with Outputs/, so they unpack straight into the thesis repository.
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
LOGS="$OUT_DIR/TUNING_LUO/logs"
mkdir -p "$LOGS"

if python "$CINECA_DIR/aggregate.py" > "$LOGS/aggregate_checks.txt" 2>&1; then
    echo "aggregate: all checks passed"
else
    echo "!!! aggregate reported FAILED CHECKS (Outputs/TUNING_LUO/logs/aggregate_checks.txt): packing anyway, send it all"
fi
tail -n 30 "$LOGS/aggregate_checks.txt"
if compgen -G "$CODE_DIR/logs/*" > /dev/null; then cp -p "$CODE_DIR"/logs/* "$LOGS/"; fi
if [ -f "$ENV_DIR/pip_freeze.txt" ]; then cp "$ENV_DIR/pip_freeze.txt" "$OUT_DIR/TUNING_LUO/pip_freeze.txt"; fi

DEST="$(dirname "$OUT_DIR")"
cd "$OUT_DIR"
ROOTS=$(ls -d TUNING_LUO FINAL_LUO 2>/dev/null || true)
LIST="$(mktemp)"
trap 'rm -f "$LIST"' EXIT

find $ROOTS -type f \( -name '*.json' -o -name '*.png' -o -name '*.txt' -o -name '*.out' -o -name '*.err' \
     -o \( -name '*.csv' ! -name 'predictions.csv' \) \) ! -name '*.tmp*' | sort > "$LIST"
tar -czf "$DEST/results_small.tar.gz" --transform 's,^,Outputs/,' -T "$LIST"
echo "results_small.tar.gz        $(du -h "$DEST/results_small.tar.gz" | cut -f1)  ($(wc -l < "$LIST") files)"

find $ROOTS -type f -name 'predictions.csv' | sort > "$LIST"
if [ -s "$LIST" ]; then
    tar -cf - --transform 's,^,Outputs/,' -T "$LIST" | gzip -1 > "$DEST/results_predictions.tar.gz"
    echo "results_predictions.tar.gz  $(du -h "$DEST/results_predictions.tar.gz" | cut -f1)  ($(wc -l < "$LIST") files)"
else
    echo "results_predictions.tar.gz  (no predictions.csv yet: the final step writes them)"
fi

find $ROOTS -type f \( -name 'best.pt' -o -name 'final.pt' \) | sort > "$LIST"
if [ -s "$LIST" ]; then
    tar -cf "$DEST/results_models.tar" --transform 's,^,Outputs/,' -T "$LIST"
    echo "results_models.tar          $(du -h "$DEST/results_models.tar" | cut -f1)  ($(wc -l < "$LIST") files)"
else
    echo "results_models.tar          (no checkpoints yet)"
fi
echo "archives in $DEST"
