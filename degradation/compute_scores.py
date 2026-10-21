from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from data_utils.manifest import read_manifest
from degradation.metrics import apply_degradation_normalization, compute_raw_metrics


SCORE_COLUMNS = [
    "image_path",
    "split",
    "original_split",
    "degradation_type",
    "sequence_id",
    "frame_id",
    "laplacian_variance",
    "blur_score",
    "saturated_pixel_ratio",
    "near_white_pixel_ratio",
    "overexposure_raw",
    "overexposure_score",
    "contrast",
    "entropy",
    "center_roi_overexposure_score",
]


def compute_scores(data: Path, limit: int = 0) -> List[Dict[str, float | str]]:
    rows = read_manifest(data / "manifest.csv")
    if limit and limit > 0:
        rows = rows[:limit]
    raw_rows: List[Dict[str, float | str]] = []
    for row in rows:
        metrics = compute_raw_metrics(row.image_path)
        raw_rows.append(
            {
                "image_path": row.image_path,
                "split": row.split,
                "original_split": row.original_split,
                "degradation_type": row.degradation_type,
                "sequence_id": row.sequence_id,
                "frame_id": row.frame_id,
                **metrics,
            }
        )
    return apply_degradation_normalization(raw_rows, train_split="train_core")


def write_scores(path: Path, rows: Sequence[Dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SCORE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in SCORE_COLUMNS})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute AgentCam degradation scores.")
    parser.add_argument("--data", type=Path, default=Path("data/processed/agentcam"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/scores/degradation_scores.csv"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry_run", "--dry-run", action="store_true", dest="dry_run")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    rows = compute_scores(args.data, limit=args.limit)
    if args.dry_run:
        print(f"Computed {len(rows)} degradation score rows.")
        if rows:
            print(rows[0])
    else:
        write_scores(args.out, rows)
        print(f"Wrote degradation scores to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
