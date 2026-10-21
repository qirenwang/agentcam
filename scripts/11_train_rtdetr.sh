#!/usr/bin/env bash
set -euo pipefail

DATA="data/processed/agentcam/data_trainval.yaml"
MODEL="rtdetr-l.pt"
PROJECT="artifacts/runs"
NAME="rtdetr_agentcam"
EPOCHS="50"
IMGSZ="640"
DRY_RUN=0
LIMIT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry_run|--dry-run) DRY_RUN=1; shift ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --data) DATA="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --project) PROJECT="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --imgsz) IMGSZ="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

CMD=(yolo detect train "model=$MODEL" "data=$DATA" "project=$PROJECT" "name=$NAME" "epochs=$EPOCHS" "imgsz=$IMGSZ")
if [[ -n "$LIMIT" ]]; then
  echo "Note: Ultralytics RT-DETR training does not support a generic --limit; use a small dataset split for smoke tests."
fi

if [[ "$DRY_RUN" == "1" ]]; then
  printf 'Dry run:'
  printf ' %q' "${CMD[@]}"
  printf '\n'
  exit 0
fi

"${CMD[@]}"
