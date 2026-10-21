from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


EnhancementKind = Literal["exposure", "sharpen"]
ActionName = Literal[
    "no_enhancement",
    "blur_enhancement",
    "overexposure_enhancement",
    "underexposure_enhancement",
    "overexposure_then_blur_enhancement",
    "defer_no_automatic_enhancement",
]

EXPOSURE_PARAMETERS = {
    "autocontrast_cutoff": 1,
    "brightness_factor": 0.92,
    "contrast_factor": 1.12,
    "gamma": 1.12,
}

SHARPEN_PARAMETERS = {
    "unsharp_mask_radius": 1.4,
    "unsharp_mask_percent": 140,
    "unsharp_mask_threshold": 3,
}

UNDEREXPOSURE_PARAMETERS = {
    "gamma": 0.72,
    "clahe_clip_limit": 2.0,
    "clahe_tile_grid_size": [8, 8],
    "clahe_color_space": "LAB",
    "clahe_channel": "L",
}

SUPPORTED_ACTIONS: tuple[ActionName, ...] = (
    "no_enhancement",
    "blur_enhancement",
    "overexposure_enhancement",
    "underexposure_enhancement",
    "overexposure_then_blur_enhancement",
    "defer_no_automatic_enhancement",
)
DEPRECATED_ACTIONS = frozenset({"mixed_sequential_or_best_of_bank"})


def exposure_enhance(image: Image.Image) -> Image.Image:
    rgb = image.convert("RGB")
    adjusted = ImageOps.autocontrast(rgb, cutoff=EXPOSURE_PARAMETERS["autocontrast_cutoff"])
    adjusted = ImageEnhance.Brightness(adjusted).enhance(EXPOSURE_PARAMETERS["brightness_factor"])
    adjusted = ImageEnhance.Contrast(adjusted).enhance(EXPOSURE_PARAMETERS["contrast_factor"])
    gamma = EXPOSURE_PARAMETERS["gamma"]
    table = [min(255, int(((value / 255.0) ** gamma) * 255.0)) for value in range(256)]
    return adjusted.point(table * 3)


def sharpen(image: Image.Image) -> Image.Image:
    rgb = image.convert("RGB")
    return rgb.filter(
        ImageFilter.UnsharpMask(
            radius=SHARPEN_PARAMETERS["unsharp_mask_radius"],
            percent=SHARPEN_PARAMETERS["unsharp_mask_percent"],
            threshold=SHARPEN_PARAMETERS["unsharp_mask_threshold"],
        )
    )


def underexposure_enhance(image: Image.Image) -> Image.Image:
    import cv2
    import numpy as np

    rgb = image.convert("RGB")
    arr = np.asarray(rgb, dtype=np.uint8)
    gamma = float(UNDEREXPOSURE_PARAMETERS["gamma"])
    lut = np.array([min(255, int(round(((value / 255.0) ** gamma) * 255.0))) for value in range(256)], dtype=np.uint8)
    gamma_corrected = cv2.LUT(arr, lut)
    lab = cv2.cvtColor(gamma_corrected, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=float(UNDEREXPOSURE_PARAMETERS["clahe_clip_limit"]),
        tileGridSize=tuple(UNDEREXPOSURE_PARAMETERS["clahe_tile_grid_size"]),
    )
    enhanced_l = clahe.apply(l_channel)
    enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(enhanced_rgb, mode="RGB")


def no_enhancement(image: Image.Image) -> Image.Image:
    return image.copy()


def operation_order_for_action(action: str) -> list[str]:
    if action == "no_enhancement":
        return ["no_enhancement"]
    if action == "blur_enhancement":
        return ["blur_enhancement"]
    if action == "overexposure_enhancement":
        return ["overexposure_enhancement"]
    if action == "underexposure_enhancement":
        return ["underexposure_gamma", "underexposure_clahe"]
    if action == "overexposure_then_blur_enhancement":
        return ["overexposure_enhancement", "blur_enhancement"]
    if action == "defer_no_automatic_enhancement":
        return []
    raise ValueError(f"Unsupported enhancement action: {action}")


def parameters_for_action(action: str) -> dict[str, Any]:
    if action == "no_enhancement":
        return {}
    if action == "blur_enhancement":
        return dict(SHARPEN_PARAMETERS)
    if action == "overexposure_enhancement":
        return dict(EXPOSURE_PARAMETERS)
    if action == "underexposure_enhancement":
        return dict(UNDEREXPOSURE_PARAMETERS)
    if action == "overexposure_then_blur_enhancement":
        return {
            "overexposure_enhancement": dict(EXPOSURE_PARAMETERS),
            "blur_enhancement": dict(SHARPEN_PARAMETERS),
        }
    if action == "defer_no_automatic_enhancement":
        return {}
    raise ValueError(f"Unsupported enhancement action: {action}")


def apply_enhancement_action(
    image: Image.Image,
    action: ActionName | str,
    operation_log: list[str] | None = None,
) -> Image.Image | None:
    if action == "no_enhancement":
        if operation_log is not None:
            operation_log.append("no_enhancement")
        return no_enhancement(image)
    if action == "blur_enhancement":
        if operation_log is not None:
            operation_log.append("blur_enhancement")
        return sharpen(image)
    if action == "overexposure_enhancement":
        if operation_log is not None:
            operation_log.append("overexposure_enhancement")
        return exposure_enhance(image)
    if action == "underexposure_enhancement":
        if operation_log is not None:
            operation_log.extend(["underexposure_gamma", "underexposure_clahe"])
        return underexposure_enhance(image)
    if action == "overexposure_then_blur_enhancement":
        if operation_log is not None:
            operation_log.append("overexposure_enhancement")
        exposure_output = exposure_enhance(image)
        if operation_log is not None:
            operation_log.append("blur_enhancement")
        return sharpen(exposure_output)
    if action == "defer_no_automatic_enhancement":
        return None
    raise ValueError(f"Unsupported enhancement action: {action}")


def enhanced_cache_path(image_path: Path, cache_dir: Path, kind: EnhancementKind) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(str(image_path.resolve()).encode("utf-8")).hexdigest()[:12]
    return cache_dir / f"{image_path.stem}_{digest}.{kind}{image_path.suffix or '.jpg'}"


def enhance_to_cache(image_path: str | Path, kind: EnhancementKind, cache_dir: str | Path = "artifacts/cache") -> Path:
    src = Path(image_path)
    dst = enhanced_cache_path(src, Path(cache_dir), kind)
    with Image.open(src) as image:
        if kind == "exposure":
            enhanced = exposure_enhance(image)
        elif kind == "sharpen":
            enhanced = sharpen(image)
        else:
            raise ValueError(f"Unsupported enhancement kind: {kind}")
        enhanced.save(dst)
    return dst
