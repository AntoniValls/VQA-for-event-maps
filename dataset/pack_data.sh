#!/usr/bin/env bash
# Packs the dataset into .tar.gz archives whose paths start with data/,
# so they can always be extracted with `tar -xzf <archive>` at the repository root.
#
#   dataset/pack_data.sh dataset               -> VQA-dataset-<date>.tar.gz  (everything except raw .svo2 recordings)
#   dataset/pack_data.sh sequence Asia Hanoi   -> Asia_Hanoi.tar.gz          (one sequence, to hand in new data)
#   dataset/pack_data.sh svo                   -> VQA-raw-svo.tar.gz         (raw ZED recordings of Barcelona)
#
# Archives are written to the current directory.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${VQA_DATA_DIR:-}"
if [[ -z "$DATA_DIR" && -f "$ROOT/.env" ]]; then
    DATA_DIR="$(sed -n 's/^VQA_DATA_DIR=//p' "$ROOT/.env" | tr -d "\"'")"
fi
DATA_DIR="${DATA_DIR:-$ROOT/data}"
[[ -d "$DATA_DIR" ]] || { echo "Data folder not found: $DATA_DIR" >&2; exit 1; }

case "${1:-}" in
    dataset)
        out="VQA-dataset-$(date +%Y-%m-%d).tar.gz"
        tar -C "$DATA_DIR" --exclude='./Europe/Barcelona/svo' --transform 's,^\.,data,' -czf "$out" .
        ;;
    sequence)
        continent="${2:?continent missing}"; city="${3:?city missing}"
        [[ -d "$DATA_DIR/$continent/$city" ]] || { echo "Not found: $DATA_DIR/$continent/$city" >&2; exit 1; }
        [[ -f "$DATA_DIR/$continent/$city/ground_truth_labels.jsonl" ]] || echo "WARNING: $continent/$city has no ground_truth_labels.jsonl yet" >&2
        out="${continent}_${city}.tar.gz"
        tar -C "$DATA_DIR" --transform 's,^,data/,' -czf "$out" "$continent/$city"
        ;;
    svo)
        out="VQA-raw-svo.tar.gz"
        tar -C "$DATA_DIR" --transform 's,^,data/,' -czf "$out" "Europe/Barcelona/svo"
        ;;
    *)
        sed -n '2,9p' "$0"; exit 1
        ;;
esac

echo "Created $out ($(du -h "$out" | cut -f1))"
