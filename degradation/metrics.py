from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
from PIL import Image, ImageStat


@dataclass
class RobustStats:
    q25: float
    median: float
    q75: float

    @property
    def iqr(self) -> float:
        return max(self.q75 - self.q25, 1e-9)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def quantile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("Cannot compute quantile of an empty sequence.")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return ordered[lower]
    weight = pos - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def fit_robust_stats(values: Iterable[float]) -> RobustStats:
    value_list = [float(value) for value in values]
    if not value_list:
        raise ValueError("Robust normalization requires at least one training value.")
    return RobustStats(
        q25=quantile(value_list, 0.25),
        median=quantile(value_list, 0.50),
        q75=quantile(value_list, 0.75),
    )


def robust_score_larger_is_worse(value: float, stats: RobustStats) -> float:
    return clamp01(0.5 + 0.25 * ((value - stats.median) / stats.iqr))


def robust_score_smaller_is_worse(value: float, stats: RobustStats) -> float:
    return clamp01(0.5 + 0.25 * ((stats.median - value) / stats.iqr))


def load_rgb(path: Path, max_side: int = 960) -> Image.Image:
    image = Image.open(path).convert("RGB")
    longest = max(image.size)
    if longest > max_side:
        scale = max_side / float(longest)
        new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        image = image.resize(new_size, Image.Resampling.BILINEAR)
    return image


def laplacian_variance(gray: Image.Image) -> float:
    if gray.width < 3 or gray.height < 3:
        return 0.0
    values = np.asarray(gray, dtype=np.float64)
    response = (
        4.0 * values[1:-1, 1:-1]
        - values[1:-1, :-2]
        - values[1:-1, 2:]
        - values[:-2, 1:-1]
        - values[2:, 1:-1]
    )
    return float(response.var())


def pixel_ratios(rgb: Image.Image) -> Dict[str, float]:
    values = np.asarray(rgb, dtype=np.uint8)
    total = max(1, values.shape[0] * values.shape[1])
    saturated = int(np.count_nonzero(values.max(axis=2) >= 250))
    near_white = int(np.count_nonzero(np.all(values >= 245, axis=2)))
    return {
        "saturated_pixel_ratio": saturated / total,
        "near_white_pixel_ratio": near_white / total,
    }


def center_roi_overexposure_score(rgb: Image.Image) -> float:
    left = rgb.width // 4
    upper = rgb.height // 4
    right = rgb.width - left
    lower = rgb.height - upper
    roi = rgb.crop((left, upper, right, lower))
    return pixel_ratios(roi)["near_white_pixel_ratio"]


def contrast(gray: Image.Image) -> float:
    return float(ImageStat.Stat(gray).stddev[0])


def entropy(gray: Image.Image) -> float:
    return float(gray.entropy())


def compute_raw_metrics(image_path: Path | str) -> Dict[str, float]:
    rgb = load_rgb(Path(image_path))
    gray = rgb.convert("L")
    ratios = pixel_ratios(rgb)
    center_score = center_roi_overexposure_score(rgb)
    overexposure_raw = max(
        ratios["saturated_pixel_ratio"],
        ratios["near_white_pixel_ratio"],
        center_score,
    )
    return {
        "laplacian_variance": laplacian_variance(gray),
        "saturated_pixel_ratio": ratios["saturated_pixel_ratio"],
        "near_white_pixel_ratio": ratios["near_white_pixel_ratio"],
        "overexposure_raw": overexposure_raw,
        "contrast": contrast(gray),
        "entropy": entropy(gray),
        "center_roi_overexposure_score": center_score,
    }


def apply_degradation_normalization(
    raw_rows: Sequence[Dict[str, float | str]],
    train_split: str = "train_core",
) -> List[Dict[str, float | str]]:
    train_rows = [row for row in raw_rows if row.get("split") == train_split]
    if not train_rows:
        raise ValueError(f"No {train_split} rows available for train-only degradation normalization.")
    lap_stats = fit_robust_stats(float(row["laplacian_variance"]) for row in train_rows)
    over_stats = fit_robust_stats(float(row["overexposure_raw"]) for row in train_rows)

    normalized = []
    for row in raw_rows:
        out = dict(row)
        out["blur_score"] = robust_score_smaller_is_worse(float(row["laplacian_variance"]), lap_stats)
        out["overexposure_score"] = robust_score_larger_is_worse(float(row["overexposure_raw"]), over_stats)
        normalized.append(out)
    return normalized
