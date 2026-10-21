#!/usr/bin/env bash
set -euo pipefail

DATA="data/processed/agentcam"
SPLIT="test"
YOLO_WEIGHTS="artifacts/runs/yolov8_agentcam/weights/best.pt"
RTDETR_WEIGHTS="weights/rtdetr_agentcam.pt"
DRY_RUN=0
LIMIT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry_run|--dry-run) DRY_RUN=1; shift ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --data) DATA="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --yolo_weights) YOLO_WEIGHTS="$2"; shift 2 ;;
    --rtdetr_weights) RTDETR_WEIGHTS="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

ARGS=(--data "$DATA" --split "$SPLIT")
if [[ -n "$LIMIT" ]]; then
  ARGS+=(--limit "$LIMIT")
fi

run_cmd() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'Dry run:'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

run_cmd python3 -m evaluation.run_detector "${ARGS[@]}" \
  --detector yolo \
  --weights "$YOLO_WEIGHTS" \
  --out artifacts/predictions/yolo_${SPLIT}.json

run_cmd python3 -m evaluation.run_detector "${ARGS[@]}" \
  --detector rtdetr \
  --weights "$RTDETR_WEIGHTS" \
  --out artifacts/predictions/rtdetr_${SPLIT}.json \
  --disable_missing_rtdetr

run_cmd python3 -m evaluation.run_detector "${ARGS[@]}" \
  --detector yolo \
  --enhancement exposure \
  --weights "$YOLO_WEIGHTS" \
  --out artifacts/predictions/yolo_exposure_${SPLIT}.json

run_cmd python3 -m evaluation.run_detector "${ARGS[@]}" \
  --detector yolo \
  --enhancement sharpen \
  --weights "$YOLO_WEIGHTS" \
  --out artifacts/predictions/yolo_sharpen_${SPLIT}.json

run_cmd python3 -m evaluation.run_detector "${ARGS[@]}" \
  --detector yolo_plus_rtdetr \
  --weights "$YOLO_WEIGHTS" \
  --rtdetr_weights "$RTDETR_WEIGHTS" \
  --out artifacts/predictions/yolo_plus_rtdetr_${SPLIT}.json \
  --disable_missing_rtdetr
