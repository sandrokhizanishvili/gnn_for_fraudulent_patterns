#!/bin/bash
# Usage (on Sandro's laptop, git-bash or WSL, from the repo): bash cineca/make_handoff.sh [output folder]
# Builds handoff_loris.zip for Loris (the file behind the Google Drive link), outside the repository (default
# ../handoff_cineca/), and prints its size. It holds README_LORIS.md (to read first) and two archives:
#   handoff_code.tar.gz  gnn_core.py + cineca/ (minus the two laptop scripts) + HANDOFF_COMMIT.txt + reference_v1/
#                        (results.json of the 8 finished v1 runs of configs 2 and 4, for aggregate's trial-0 print)
#   handoff_data.tar.gz  exactly the files gnn_core.load_graphs reads for GFP v0 / no node encoding:
#                        Data/train_graph.pt, val_graph.pt, test_graph.pt, feature_meta.json + Data/SHA256SUMS
# Not shipped to Cineca. Commit first, so HANDOFF_COMMIT.txt names the code Loris runs.
set -eo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$REPO/../handoff_cineca}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
cd "$REPO"

DIRTY="$(git status --porcelain -- gnn_core.py cineca)"
if [ -n "$DIRTY" ]; then
    echo "WARNING: uncommitted changes in gnn_core.py / cineca/ (commit first so HANDOFF_COMMIT.txt is exact):"
    echo "$DIRTY"
fi

# ---- code -------------------------------------------------------------------------
mkdir -p "$STAGE/code/cineca"
cp gnn_core.py "$STAGE/code/"
for f in cineca/*; do
    case "$(basename "$f")" in
        make_handoff.sh|make_env_pack.sh|__pycache__|logs) ;;
        *) cp -r "$f" "$STAGE/code/cineca/" ;;
    esac
done
for op in gin pna gat transformer; do
    fam="$(echo "$op" | tr '[:lower:]' '[:upper:]')"
    for run in "${op}_mp-base_readout-base_dir-in" "${op}_mp-full_readout-full_dir-in"; do
        mkdir -p "$STAGE/code/reference_v1/$fam/$run"
        cp "Outputs/$fam/$run/results.json" "$STAGE/code/reference_v1/$fam/$run/"
    done
done
{
    echo "commit $(git rev-parse HEAD)"
    echo "built  $(date '+%Y-%m-%d %H:%M')"
    if [ -n "$DIRTY" ]; then echo "$DIRTY" | sed 's/^/uncommitted: /'; fi
} > "$STAGE/code/HANDOFF_COMMIT.txt"
# Linux line endings in every text file (core.autocrlf=true checks them out with CRLF on Windows)
find "$STAGE/code" -type f \( -name '*.sh' -o -name '*.sbatch' -o -name '*.py' -o -name '*.md' -o -name '*.lock' \
     -o -name '*.json' -o -name '*.txt' \) -exec sed -i 's/\r$//' {} +
if grep -rlI $'\r' "$STAGE/code" > /dev/null; then echo "CR characters left in the code archive" >&2; exit 1; fi
tar -C "$STAGE/code" -czf "$OUT/handoff_code.tar.gz" .
echo "handoff_code.tar.gz  $(du -h "$OUT/handoff_code.tar.gz" | cut -f1)  ($(tar -tzf "$OUT/handoff_code.tar.gz" | grep -vc '/$') files)"

# ---- data -------------------------------------------------------------------------
FILES="train_graph.pt val_graph.pt test_graph.pt feature_meta.json"
(cd Data && sha256sum $FILES > "$STAGE/SHA256SUMS")
tar -czf "$OUT/handoff_data.tar.gz" $(for f in $FILES; do echo "Data/$f"; done) \
    -C "$STAGE" --transform 's,^SHA256SUMS$,Data/SHA256SUMS,' SHA256SUMS
echo "handoff_data.tar.gz  $(du -h "$OUT/handoff_data.tar.gz" | cut -f1)  (unpacked: $(cd Data && du -ch $FILES | tail -1 | cut -f1))"

# ---- one zip for the Drive link (stored, not compressed: the archives are gzipped already) ----------------
sed 's/\r$//' cineca/README_LORIS.md > "$STAGE/README_LORIS.md"
PY="$(command -v python || command -v python3)"
rm -f "$OUT/handoff_loris.zip"
"$PY" -c 'import os, sys, zipfile
with zipfile.ZipFile(sys.argv[1], "w", zipfile.ZIP_STORED) as z:
    for f in sys.argv[2:]:
        z.write(f, os.path.basename(f))' \
    "$OUT/handoff_loris.zip" "$STAGE/README_LORIS.md" "$OUT/handoff_code.tar.gz" "$OUT/handoff_data.tar.gz"
rm "$OUT/handoff_code.tar.gz" "$OUT/handoff_data.tar.gz"
echo "handoff_loris.zip    $(du -h "$OUT/handoff_loris.zip" | cut -f1)  (README_LORIS.md + the two archives) in $OUT"
