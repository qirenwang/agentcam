from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal


CanonicalVlmClass = Literal[
    "blur",
    "overexposure",
    "underexposure",
    "blur_overexposure",
    "clear",
    "unknown",
]

EnhancementAction = Literal[
    "no_enhancement",
    "blur_enhancement",
    "overexposure_enhancement",
    "underexposure_enhancement",
    "overexposure_then_blur_enhancement",
    "defer_no_automatic_enhancement",
]

CANONICAL_VLM_CLASSES: tuple[CanonicalVlmClass, ...] = (
    "blur",
    "overexposure",
    "underexposure",
    "blur_overexposure",
    "clear",
    "unknown",
)

ACTION_BY_CANONICAL_CLASS: dict[CanonicalVlmClass, EnhancementAction] = {
    "clear": "no_enhancement",
    "blur": "blur_enhancement",
    "overexposure": "overexposure_enhancement",
    "underexposure": "underexposure_enhancement",
    "blur_overexposure": "overexposure_then_blur_enhancement",
    "unknown": "defer_no_automatic_enhancement",
}

DEPRECATED_C1_ACTIONS: frozenset[str] = frozenset({"mixed_sequential_or_best_of_bank"})


def _payload(record: Mapping[str, Any]) -> Mapping[str, Any]:
    parsed_json = record.get("parsed_json")
    if isinstance(parsed_json, Mapping) and not all(
        field in record for field in ("blur", "overexposure", "underexposure", "clear")
    ):
        return parsed_json
    return record


def _strict_bool(payload: Mapping[str, Any], field: str) -> bool | None:
    value = payload.get(field)
    if isinstance(value, bool):
        return value
    return None


def canonicalize_vlm_degradation(record: Mapping[str, Any]) -> CanonicalVlmClass:
    """Collapse the C1 multi-boolean VLM schema to the frozen C2 class set."""
    payload = _payload(record)
    blur = _strict_bool(payload, "blur")
    overexposure = _strict_bool(payload, "overexposure")
    underexposure = _strict_bool(payload, "underexposure")
    clear = _strict_bool(payload, "clear")
    unknown = _strict_bool(payload, "unknown_degradation")
    if unknown is None:
        unknown = _strict_bool(payload, "unknown")

    if None in {blur, overexposure, underexposure, clear, unknown}:
        return "unknown"

    if unknown:
        return "unknown"

    degradation_flags = (blur, overexposure, underexposure)
    if clear and any(degradation_flags):
        return "unknown"
    if clear and not any(degradation_flags):
        return "clear"

    if overexposure and underexposure:
        return "unknown"
    if blur and underexposure:
        return "unknown"
    if blur and overexposure and not underexposure:
        return "blur_overexposure"
    if blur and not overexposure and not underexposure:
        return "blur"
    if overexposure and not blur and not underexposure:
        return "overexposure"
    if underexposure and not blur and not overexposure:
        return "underexposure"
    return "unknown"


def action_for_canonical_class(canonical_class: CanonicalVlmClass | str) -> EnhancementAction:
    """Map one frozen canonical VLM class to one deterministic C2 action."""
    try:
        return ACTION_BY_CANONICAL_CLASS[canonical_class]  # type: ignore[index]
    except KeyError as exc:
        raise ValueError(f"Unsupported canonical VLM class: {canonical_class!r}") from exc


def action_from_vlm_record(record: Mapping[str, Any]) -> EnhancementAction:
    return action_for_canonical_class(canonicalize_vlm_degradation(record))
