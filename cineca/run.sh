#!/bin/bash
# Usage: bash cineca/run.sh <step> [operator] [config] [--first N]
#   check      environment test on a GPU node (debug queue, ~5 min)
#   probe      which depths fit on the GPU, per operator (4 GPU tasks, < 8 h)
#   draw       the pre-registered trial list (login node, seconds)
#   tune       every unfinished (trial, seed) of every (operator, config): 8 job arrays;
#              --first 10 = only trials 0-9 (the early check-in); run it again daily until status shows all done
#   aggregate  correctness checks, the winner of each (operator, config), summaries (login node, minutes)
#   final      the 5-seed retrain of each winner: 8 job arrays of 5 tasks
#   status     done / running / queued / timeout / failed / missing per step and (operator, config)
#   pack       the aggregate checks + the three result archives to send back
# operator / config restrict tune and final to one pair, e.g. "bash cineca/run.sh tune pna 4".
# Re-running a step only submits what is unfinished and not already queued or running.
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
mkdir -p "$CODE_DIR/logs"

STEP="${1:?usage: bash cineca/run.sh <check|probe|draw|tune|aggregate|final|status|pack> [operator] [config] [--first N]}"
shift
OPS=(gin pna gat transformer)
CFGS=(2 4)
FIRST=""
POSITIONAL=()
while [ $# -gt 0 ]; do
    case "$1" in
        --first) FIRST="$2"; shift 2 ;;
        *) POSITIONAL+=("$1"); shift ;;
    esac
done
if [ ${#POSITIONAL[@]} -ge 1 ]; then OPS=("${POSITIONAL[0]}"); fi
if [ ${#POSITIONAL[@]} -ge 2 ]; then CFGS=("${POSITIONAL[1]}"); fi

# submit <job name> <array spec or ""> <time> <memory> <qos> [VAR=value ...]
submit() {
    local name="$1" array="$2" time="$3" mem="$4" qos="$5"
    shift 5
    local exports="ALL,CODE_DIR=$CODE_DIR,STEP=$STEP"
    for kv in "$@"; do exports="$exports,$kv"; done
    local args=(--account="$ACCOUNT" --partition="$PARTITION" --qos="$qos" --time="$time" --mem="$mem"
                --cpus-per-task="$CPUS" --gres=gpu:1 --chdir="$CODE_DIR" --job-name="$name" --export="$exports")
    if [ -n "$array" ]; then
        args+=(--array="${array}%${MAX_PARALLEL}"
               --output="$CODE_DIR/logs/%x_%A_%a.out" --error="$CODE_DIR/logs/%x_%A_%a.err")
    else
        args+=(--output="$CODE_DIR/logs/%x_%j.out" --error="$CODE_DIR/logs/%x_%j.err")
    fi
    if sbatch "${args[@]}" "$CINECA_DIR/job.sbatch"; then
        echo "  submitted $name ${array:+array $array} (mem $mem, time $time)"
    else
        echo "  !! sbatch refused $name (for example a per-user job limit): nothing is lost, run this step again later"
    fi
}

# submit_plan <time>: one job array per line of "luo.py plan"
submit_plan() {
    local plan
    plan=$(python "$CINECA_DIR/luo.py" plan "$STEP" --ops "${OPS[@]}" --cfgs "${CFGS[@]}" ${FIRST:+--first "$FIRST"})
    while IFS='|' read -r name ids mem op cfg; do
        [ -z "$name" ] && continue
        if [ "$STEP" = probe ]; then
            submit "$name" "$ids" "$1" "$mem" "$QOS"
        else
            submit "$name" "$ids" "$1" "$mem" "$QOS" "OPERATOR=$op" "CONFIG=$cfg"
        fi
    done <<< "$plan"
}

# pack: the aggregate checks, then three archives next to OUT_DIR (paths start with Outputs/, so they unpack
# straight into the thesis repository): results_small.tar.gz (every small result file + the job logs; also the early
# check-in file), results_predictions.tar.gz (predictions.csv of the final retrain), results_models.tar (the kept
# checkpoints; not gzipped, weights barely compress)
pack() {
    local logs="$OUT_DIR/TUNING_LUO/logs" dest list roots
    mkdir -p "$logs"
    if python "$CINECA_DIR/luo.py" aggregate > "$logs/aggregate_checks.txt" 2>&1; then
        echo "aggregate: all checks passed"
    else
        echo "!!! aggregate reported FAILED CHECKS (Outputs/TUNING_LUO/logs/aggregate_checks.txt): packing anyway, send it all"
    fi
    tail -n 30 "$logs/aggregate_checks.txt"
    if compgen -G "$CODE_DIR/logs/*" > /dev/null; then cp -p "$CODE_DIR"/logs/* "$logs/"; fi
    dest="$(dirname "$OUT_DIR")"
    cd "$OUT_DIR"
    roots=$(ls -d TUNING_LUO FINAL_LUO 2>/dev/null || true)
    list="$(mktemp)"
    trap 'rm -f "$list"' EXIT
    find $roots -type f \( -name '*.json' -o -name '*.png' -o -name '*.txt' -o -name '*.out' -o -name '*.err' \
         -o \( -name '*.csv' ! -name 'predictions.csv' \) \) ! -name '*.tmp*' | sort > "$list"
    tar -czf "$dest/results_small.tar.gz" --transform 's,^,Outputs/,' -T "$list"
    echo "results_small.tar.gz        $(du -h "$dest/results_small.tar.gz" | cut -f1)  ($(wc -l < "$list") files)"
    find $roots -type f -name 'predictions.csv' | sort > "$list"
    if [ -s "$list" ]; then
        tar -cf - --transform 's,^,Outputs/,' -T "$list" | gzip -1 > "$dest/results_predictions.tar.gz"
        echo "results_predictions.tar.gz  $(du -h "$dest/results_predictions.tar.gz" | cut -f1)  ($(wc -l < "$list") files)"
    else
        echo "results_predictions.tar.gz  (no predictions.csv yet: the final step writes them)"
    fi
    find $roots -type f \( -name 'best.pt' -o -name 'final.pt' \) | sort > "$list"
    if [ -s "$list" ]; then
        tar -cf "$dest/results_models.tar" --transform 's,^,Outputs/,' -T "$list"
        echo "results_models.tar          $(du -h "$dest/results_models.tar" | cut -f1)  ($(wc -l < "$list") files)"
    else
        echo "results_models.tar          (no checkpoints yet)"
    fi
    echo "archives in $dest"
}

case "$STEP" in
    check)     submit check "" "$TIME_CHECK" 32G "$QOS_DEBUG" ;;
    probe)     submit_plan "$TIME_PROBE" ;;
    draw)      python "$CINECA_DIR/luo.py" draw ;;
    tune)      submit_plan "$TIME_TUNE" ;;
    aggregate) python "$CINECA_DIR/luo.py" aggregate ;;
    final)     submit_plan "$TIME_FINAL" ;;
    status)    python "$CINECA_DIR/luo.py" status ;;
    pack)      pack ;;
    *)         echo "unknown step '$STEP': use check, probe, draw, tune, aggregate, final, status or pack" >&2; exit 1 ;;
esac
