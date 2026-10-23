#!/bin/bash
# Usage: bash cineca/run_step.sh <step> [operator] [config] [--first N]
#   check      environment test on a GPU node (debug queue, ~5 min)
#   probe      which depths fit on the GPU, per operator (4 GPU tasks, < 8 h)
#   draw       the pre-registered trial list (login node, seconds)
#   tune       every unfinished (trial, seed) of every (operator, config): 8 job arrays;
#              --first 10 = only trials 0-9 (the early check-in); run it again daily until status.sh shows all done
#   aggregate  correctness checks, the winner of each (operator, config), summaries (login node, minutes)
#   final      the 5-seed retrain of each winner: 8 job arrays of 5 tasks
# operator / config restrict tune and final to one pair, e.g. "bash cineca/run_step.sh tune pna 4".
# Re-running a step only submits what is unfinished and not already queued or running.
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
mkdir -p "$CODE_DIR/logs"

STEP="${1:?usage: bash cineca/run_step.sh <check|probe|draw|tune|aggregate|final> [operator] [config] [--first N]}"
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

# submit <job name> <array spec or ""> <time> <memory> <qos> <sbatch file> [VAR=value ...]
submit() {
    local name="$1" array="$2" time="$3" mem="$4" qos="$5" file="$6"
    shift 6
    local exports="ALL,CODE_DIR=$CODE_DIR"
    for kv in "$@"; do exports="$exports,$kv"; done
    local args=(--account="$ACCOUNT" --partition="$PARTITION" --qos="$qos" --time="$time" --mem="$mem"
                --cpus-per-task="$CPUS" --gres=gpu:1 --chdir="$CODE_DIR" --job-name="$name" --export="$exports")
    if [ -n "$array" ]; then
        args+=(--array="${array}%${MAX_PARALLEL}"
               --output="$CODE_DIR/logs/%x_%A_%a.out" --error="$CODE_DIR/logs/%x_%A_%a.err")
    else
        args+=(--output="$CODE_DIR/logs/%x_%j.out" --error="$CODE_DIR/logs/%x_%j.err")
    fi
    if sbatch "${args[@]}" "$CINECA_DIR/$file"; then
        echo "  submitted $name ${array:+array $array} (mem $mem, time $time)"
    else
        echo "  !! sbatch refused $name (for example a per-user job limit): nothing is lost, run this step again later"
    fi
}

# submit_plan <step> <sbatch file> <time>: one job array per line of "common.py plan"
submit_plan() {
    local step="$1" file="$2" time="$3"
    local plan
    plan=$(python "$CINECA_DIR/common.py" plan "$step" --ops "${OPS[@]}" --cfgs "${CFGS[@]}" ${FIRST:+--first "$FIRST"})
    while IFS='|' read -r name ids mem op cfg; do
        [ -z "$name" ] && continue
        if [ "$step" = probe ]; then
            submit "$name" "$ids" "$time" "$mem" "$QOS" "$file"
        else
            submit "$name" "$ids" "$time" "$mem" "$QOS" "$file" "OPERATOR=$op" "CONFIG=$cfg"
        fi
    done <<< "$plan"
}

case "$STEP" in
    check)     submit check "" "$TIME_CHECK" 32G "$QOS_DEBUG" check.sbatch ;;
    probe)     submit_plan probe probe.sbatch "$TIME_PROBE" ;;
    draw)      python "$CINECA_DIR/draw_trials.py" ;;
    tune)      submit_plan tune tune.sbatch "$TIME_TUNE" ;;
    aggregate) python "$CINECA_DIR/aggregate.py" ;;
    final)     submit_plan final final.sbatch "$TIME_FINAL" ;;
    *)         echo "unknown step '$STEP': use check, probe, draw, tune, aggregate or final" >&2; exit 1 ;;
esac
