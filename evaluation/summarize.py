from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from degradation.metrics import quantile
from evaluation.latency import summarize_latency_values


DETECTION_COLUMNS = [
    "method",
    "mAP50",
    "mAP50-95",
    "precision",
    "recall",
    "critical_FN_rate",
]


def parse_named_paths(items: Sequence[str]) -> List[Tuple[str, Path]]:
    parsed = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected name=path, got: {item}")
        name, path = item.split("=", 1)
        parsed.append((name, Path(path)))
    return parsed


def write_csv(path: Path, columns: Sequence[str], rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def detection_rows(metrics: Sequence[Tuple[str, Path]]) -> Tuple[List[str], List[Dict[str, object]]]:
    rows = []
    degradation_columns: List[str] = []
    for method, path in metrics:
        payload = json.loads(path.read_text())
        row: Dict[str, object] = {"method": method}
        for column in DETECTION_COLUMNS[1:]:
            row[column] = payload.get(column, "")
        for group, scoped in dict(payload.get("by_degradation", {})).items():
            column = f"{group}_mAP50"
            row[column] = scoped.get("mAP50", "")
            if column not in degradation_columns:
                degradation_columns.append(column)
        # Metric files written before by_degradation existed carry two flat keys.
        for legacy_key, column in [("blur_mAP", "blur_mAP50"), ("overexposure_mAP", "overexposure_mAP50")]:
            if legacy_key in payload and column not in row:
                row[column] = payload[legacy_key]
                if column not in degradation_columns:
                    degradation_columns.append(column)
        rows.append(row)
    return DETECTION_COLUMNS + sorted(degradation_columns), rows


def load_jsonl(path: Path, limit: int = 0) -> List[Dict[str, object]]:
    records = []
    with path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if limit and len(records) >= limit:
                break
    return records


def latency_and_routing_row(method: str, records: Sequence[Dict[str, object]]) -> Dict[str, object]:
    totals = [float(record.get("latency_ms", {}).get("total_ms", 0.0)) for record in records]
    summary = summarize_latency_values(totals)
    actions = Counter(str(record.get("selected_action", record.get("action", ""))) for record in records)
    total = max(1, len(records))
    enhancement_calls = actions["EXPOSURE_ENHANCE_THEN_YOLO"] + actions["SHARPEN_THEN_YOLO"]
    return {
        "method": method,
        "mean_latency_ms": summary["mean"],
        "p50_latency_ms": summary["p50"],
        "p95_latency_ms": summary["p95"],
        "rtdetr_call_rate": actions["RTDETR_VERIFY"] / total,
        "enhancement_call_rate": enhancement_calls / total,
        "uncertainty_signal_rate": actions["UNCERTAINTY_SIGNAL"] / total,
    }


def score_quantile_rows(scores_csv: Path) -> List[Dict[str, object]]:
    with scores_csv.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    output = []
    for field in ["blur_score", "overexposure_score"]:
        values = [float(row[field]) for row in rows if row.get(field) not in {"", None}]
        if not values:
            continue
        q33 = quantile(values, 0.33)
        q67 = quantile(values, 0.67)
        buckets = Counter()
        for row in rows:
            value = float(row[field])
            if value <= q33:
                bucket = "bottom_33"
            elif value <= q67:
                bucket = "middle_33"
            else:
                bucket = "top_33"
            buckets[bucket] += 1
        for bucket in ["bottom_33", "middle_33", "top_33"]:
            output.append({"score": field, "bucket": bucket, "count": buckets[bucket]})
    return output


def write_histogram_svg(path: Path, title: str, values: Sequence[float], bins: int = 20) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not values:
        path.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"640\" height=\"240\"></svg>\n")
        return
    min_v = min(values)
    max_v = max(values)
    width = 640
    height = 240
    margin = 32
    counts = [0 for _ in range(bins)]
    span = max(max_v - min_v, 1e-9)
    for value in values:
        idx = min(bins - 1, int(((value - min_v) / span) * bins))
        counts[idx] += 1
    max_count = max(counts) or 1
    bar_w = (width - 2 * margin) / bins
    rects = []
    for idx, count in enumerate(counts):
        bar_h = (height - 2 * margin) * count / max_count
        x = margin + idx * bar_w
        y = height - margin - bar_h
        rects.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - 2:.1f}" height="{bar_h:.1f}" fill="#2f6f73"/>')
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<text x="{margin}" y="22" font-family="sans-serif" font-size="16">{title}</text>',
        *rects,
        "</svg>",
        "",
    ]
    path.write_text("\n".join(svg))


def write_action_distribution_svg(path: Path, records: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    actions = Counter(str(record.get("selected_action", record.get("action", ""))) for record in records)
    rows = [{"action": action, "count": count} for action, count in sorted(actions.items())]
    write_csv(path.with_suffix(".csv"), ["action", "count"], rows)
    values = list(actions.values())
    write_histogram_svg(path, "Action distribution", [float(value) for value in values], bins=max(1, len(values)))


def summarize(
    metrics: Sequence[Tuple[str, Path]],
    router_log: Optional[Path],
    scores_csv: Optional[Path],
    out_dir: Path,
    figures_dir: Path,
    limit: int = 0,
) -> None:
    if metrics:
        columns, rows = detection_rows(metrics)
        write_csv(out_dir / "detection_robustness.csv", columns, rows)

    records: List[Dict[str, object]] = []
    if router_log and router_log.exists():
        records = load_jsonl(router_log, limit=limit)
        write_csv(
            out_dir / "latency_and_routing.csv",
            [
                "method",
                "mean_latency_ms",
                "p50_latency_ms",
                "p95_latency_ms",
                "rtdetr_call_rate",
                "enhancement_call_rate",
                "uncertainty_signal_rate",
            ],
            [latency_and_routing_row("AgentCam_router", records)],
        )
        write_action_distribution_svg(figures_dir / "action_distribution.svg", records)
        latencies = [float(record.get("latency_ms", {}).get("total_ms", 0.0)) for record in records]
        write_histogram_svg(figures_dir / "latency_distribution.svg", "Latency distribution", latencies)

    if scores_csv and scores_csv.exists():
        quantile_rows = score_quantile_rows(scores_csv)
        write_csv(out_dir / "score_quantiles.csv", ["score", "bucket", "count"], quantile_rows)
        with scores_csv.open("r", newline="") as f:
            rows = list(csv.DictReader(f))
        blur_values = [float(row["blur_score"]) for row in rows if row.get("blur_score")]
        over_values = [float(row["overexposure_score"]) for row in rows if row.get("overexposure_score")]
        write_histogram_svg(figures_dir / "degradation_blur_histogram.svg", "Blur score histogram", blur_values)
        write_histogram_svg(
            figures_dir / "degradation_overexposure_histogram.svg",
            "Overexposure score histogram",
            over_values,
        )

    ablation_methods = [
        "YOLO only",
        "router without degradation score",
        "router without uncertainty score",
        "router without RT-DETR",
        "full AgentCam router",
        "oracle upper bound",
    ]
    write_csv(out_dir / "router_ablation.csv", ["method", "mAP50", "mean_latency_ms"], [{"method": m} for m in ablation_methods])
    (figures_dir / "representative_failure_recovery_cases.json").write_text("[]\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize AgentCam result tables and figure artifacts.")
    parser.add_argument("--metrics", nargs="*", default=[], help="Detection metric files as method=path.")
    parser.add_argument("--router_log", type=Path)
    parser.add_argument("--scores", type=Path)
    parser.add_argument("--out_dir", type=Path, default=Path("artifacts/tables"))
    parser.add_argument("--figures_dir", type=Path, default=Path("artifacts/figures"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry_run", "--dry-run", action="store_true", dest="dry_run")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    named_metrics = parse_named_paths(args.metrics)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "metrics": [name for name, _ in named_metrics],
                    "router_log": str(args.router_log) if args.router_log else None,
                    "scores": str(args.scores) if args.scores else None,
                    "out_dir": str(args.out_dir),
                    "figures_dir": str(args.figures_dir),
                },
                indent=2,
            )
        )
    else:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        args.figures_dir.mkdir(parents=True, exist_ok=True)
        summarize(named_metrics, args.router_log, args.scores, args.out_dir, args.figures_dir, limit=args.limit)
        print(f"Wrote summary tables to {args.out_dir}")
        print(f"Wrote figure artifacts to {args.figures_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
