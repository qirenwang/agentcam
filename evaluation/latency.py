from __future__ import annotations

import argparse
import json
import statistics
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


def cuda_synchronize() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def timed_section(latencies: Dict[str, float], key: str, synchronize_cuda: bool = True):
    if synchronize_cuda:
        cuda_synchronize()
    start = time.perf_counter()
    try:
        yield
    finally:
        if synchronize_cuda:
            cuda_synchronize()
        latencies[key] = latencies.get(key, 0.0) + (time.perf_counter() - start) * 1000.0


def summarize_latency_values(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
    ordered = sorted(values)

    def percentile(p: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))
        return ordered[idx]

    return {
        "mean": statistics.mean(ordered),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "max": ordered[-1],
    }


def summarize_jsonl_latency(path: Path) -> Dict[str, Dict[str, float]]:
    values: Dict[str, List[float]] = {}
    with path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            latency = record.get("latency_ms", {})
            for key, value in latency.items():
                values.setdefault(key, []).append(float(value))
    return {key: summarize_latency_values(val) for key, val in values.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize AgentCam JSONL latency logs.")
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/tables/latency_summary.json"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry_run", "--dry-run", action="store_true", dest="dry_run")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = summarize_jsonl_latency(args.log)
    if args.dry_run:
        print(json.dumps(summary, indent=2))
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2))
        print(f"Wrote latency summary to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
