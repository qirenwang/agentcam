#!/usr/bin/env bash
set -euo pipefail

DRIVE_URL="https://drive.google.com/file/d/1ROUN1MkJ2l36KLBsEf_-BNF81xFD2uWe/view"
OUT="data/raw/data_eda_most25.zip"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry_run|--dry-run)
      DRY_RUN=1
      shift
      ;;
    --limit)
      shift 2
      ;;
    --out)
      OUT="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

mkdir -p "$(dirname "$OUT")"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry run: would download $DRIVE_URL to $OUT"
  exit 0
fi

if ! command -v gdown >/dev/null 2>&1; then
  echo "gdown is not installed. Install dependencies with: pip install -r requirements.txt" >&2
  echo "Manual fallback: download the Google Drive file and place it at $OUT" >&2
  exit 1
fi

if gdown --fuzzy "$DRIVE_URL" -O "$OUT"; then
  echo "Downloaded dataset archive to $OUT"
else
  echo "gdown failed. This script does not bypass Google Drive permissions or quota limits." >&2
  echo "Manual fallback:" >&2
  echo "  1. Open: $DRIVE_URL" >&2
  echo "  2. Download the archive." >&2
  echo "  3. Place it at: $OUT" >&2
  exit 1
fi
