from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from statistics import mean

from PIL import Image, ImageFilter

from degradation.metrics import compute_raw_metrics, laplacian_variance, pixel_ratios
from routing.rule_router import (
    EXPOSURE_ENHANCE_THEN_YOLO,
    RTDETR_VERIFY,
    SHARPEN_THEN_YOLO,
    UNCERTAINTY_SIGNAL,
    YOLO_ONLY,
    RuleRouter,
    RouterInputs,
)


def write_checkerboard(path: Path) -> None:
    image = Image.new("RGB", (64, 64))
    pixels = image.load()
    for y in range(64):
        for x in range(64):
            value = 255 if (x // 4 + y // 4) % 2 == 0 else 0
            pixels[x, y] = (value, value, value)
    image.save(path)


class DegradationAndRoutingTests(unittest.TestCase):
    def test_degradation_metrics_sanity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            sharp = tmp_path / "sharp.jpg"
            blurred = tmp_path / "blurred.jpg"
            over = tmp_path / "over.jpg"
            write_checkerboard(sharp)
            Image.open(sharp).filter(ImageFilter.GaussianBlur(radius=2)).save(blurred)
            Image.new("RGB", (64, 64), (255, 255, 255)).save(over)

            sharp_metrics = compute_raw_metrics(sharp)
            blurred_metrics = compute_raw_metrics(blurred)
            over_metrics = compute_raw_metrics(over)

            self.assertGreater(sharp_metrics["laplacian_variance"], blurred_metrics["laplacian_variance"])
            self.assertGreater(over_metrics["near_white_pixel_ratio"], 0.99)
            self.assertGreater(over_metrics["center_roi_overexposure_score"], 0.99)

    def test_vectorized_metrics_match_reference_loops(self) -> None:
        rng_state = 12345
        gray = Image.new("L", (24, 18))
        rgb = Image.new("RGB", (24, 18))
        gray_pixels = gray.load()
        rgb_pixels = rgb.load()
        for y in range(18):
            for x in range(24):
                rng_state = (1103515245 * rng_state + 12345) % (2**31)
                value = rng_state % 256
                gray_pixels[x, y] = value
                rgb_pixels[x, y] = (value, (value * 7) % 256, (value * 13) % 256)

        pixels = gray.load()
        reference_values = []
        for y in range(1, gray.height - 1):
            for x in range(1, gray.width - 1):
                reference_values.append(
                    4.0 * pixels[x, y]
                    - pixels[x - 1, y]
                    - pixels[x + 1, y]
                    - pixels[x, y - 1]
                    - pixels[x, y + 1]
                )
        avg = mean(reference_values)
        reference_variance = mean((value - avg) ** 2 for value in reference_values)
        self.assertAlmostEqual(laplacian_variance(gray), reference_variance, places=6)

        saturated = 0
        near_white = 0
        for y in range(18):
            for x in range(24):
                red, green, blue = rgb_pixels[x, y]
                if max(red, green, blue) >= 250:
                    saturated += 1
                if red >= 245 and green >= 245 and blue >= 245:
                    near_white += 1
        ratios = pixel_ratios(rgb)
        total = 24 * 18
        self.assertAlmostEqual(ratios["saturated_pixel_ratio"], saturated / total, places=9)
        self.assertAlmostEqual(ratios["near_white_pixel_ratio"], near_white / total, places=9)

    def test_rule_router_action_choices(self) -> None:
        router = RuleRouter(
            {
                "thresholds": {
                    "blur_score": 0.7,
                    "overexposure_score": 0.7,
                    "low_mean_confidence": 0.35,
                    "low_max_confidence": 0.45,
                    "low_confidence_ratio": 0.5,
                    "enhancement_min_budget_ms": 35,
                    "rtdetr_min_budget_ms": 80,
                },
                "router": {"enable_rtdetr": True, "enable_temporal_recovery": False, "default_latency_budget_ms": 100},
            }
        )

        base = dict(
            blur_score=0.1,
            overexposure_score=0.1,
            saturated_pixel_ratio=0.0,
            yolo_mean_confidence=0.8,
            yolo_max_confidence=0.9,
            num_yolo_detections=2,
            low_confidence_detection_ratio=0.0,
            latency_budget_ms=100,
        )
        self.assertEqual(router.route(RouterInputs(**base)).action, YOLO_ONLY)
        self.assertEqual(router.route(RouterInputs(**{**base, "overexposure_score": 0.9})).action, EXPOSURE_ENHANCE_THEN_YOLO)
        self.assertEqual(router.route(RouterInputs(**{**base, "blur_score": 0.9})).action, SHARPEN_THEN_YOLO)
        self.assertEqual(router.route(RouterInputs(**{**base, "yolo_mean_confidence": 0.1})).action, RTDETR_VERIFY)
        self.assertEqual(
            router.route(RouterInputs(**{**base, "yolo_mean_confidence": 0.1, "latency_budget_ms": 20})).action,
            UNCERTAINTY_SIGNAL,
        )


if __name__ == "__main__":
    unittest.main()
