"""Small, typed helpers for the public receipt-extraction response.

The take-home evaluator imports :func:`src.pipeline.extract` directly and is
deliberately strict about its return shape.  Keeping construction of that
shape here avoids slightly different failure responses from the image, OCR,
and parsing stages.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any, TypeAlias


FIELD_NAMES: tuple[str, ...] = (
    "merchant_name",
    "transaction_date",
    "total_amount",
    "currency",
)
EXPECTED_FIELDS = FIELD_NAMES

FieldValue: TypeAlias = str | float | None
FieldDict: TypeAlias = dict[str, FieldValue | float | bool]
FieldsDict: TypeAlias = dict[str, FieldDict]
ResultDict: TypeAlias = dict[str, Any]


def _confidence(value: object) -> float:
    """Return a finite native float in the public ``[0, 1]`` range."""

    if isinstance(value, bool) or not isinstance(value, Real):
        return 0.0
    numeric = float(value)
    if not math.isfinite(numeric):
        return 0.0
    return float(min(1.0, max(0.0, numeric)))


def _clean_value(value: object) -> FieldValue:
    """Keep only JSON-safe scalar values used by the four required fields."""

    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    return numeric


@dataclass(frozen=True)
class FieldResult:
    """A single field in the stable public response schema."""

    value: FieldValue = None
    confidence: float = 0.0
    needs_review: bool = True

    def as_dict(self) -> FieldDict:
        value = _clean_value(self.value)
        # A missing value is always an abstention, never an unflagged guess.
        needs_review = bool(self.needs_review) or value is None
        return {
            "value": value,
            "confidence": _confidence(self.confidence),
            "needs_review": needs_review,
        }


def make_field(
    value: FieldValue = None,
    confidence: float = 0.0,
    needs_review: bool = True,
) -> FieldDict:
    """Create a normalized field mapping using only native JSON scalars."""

    return FieldResult(value, confidence, needs_review).as_dict()


def empty_fields() -> FieldsDict:
    """Return a reviewed abstention for every required extraction field."""

    return {name: make_field() for name in FIELD_NAMES}


def _field_from_mapping(raw_field: object) -> FieldDict:
    if isinstance(raw_field, Mapping):
        return make_field(
            raw_field.get("value"),
            raw_field.get("confidence", 0.0),
            raw_field.get("needs_review", True) is True,
        )
    return make_field()


def _issues(raw_issues: Sequence[object] | object, passed: bool) -> list[str]:
    """Normalize user-facing issues and enforce the quality invariant."""

    if passed:
        return []
    if isinstance(raw_issues, str):
        candidates: Sequence[object] = (raw_issues,)
    elif isinstance(raw_issues, Sequence):
        candidates = raw_issues
    else:
        candidates = ()

    cleaned: list[str] = []
    for issue in candidates:
        text = str(issue).strip()
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned or ["The image could not be processed safely. Please retake the receipt photo."]


def make_result(
    passed: bool,
    issues: Sequence[object] | object = (),
    fields: Mapping[str, object] | None = None,
    processing_ms: float = 0.0,
) -> ResultDict:
    """Build the exact public response schema on every code path.

    A failed quality/processing stage intentionally discards all values.  This
    prevents an earlier partial OCR guess from becoming an unreviewed result.
    """

    pass_value = bool(passed)
    try:
        elapsed = float(processing_ms)
    except (TypeError, ValueError):
        elapsed = 0.0
    if not math.isfinite(elapsed) or elapsed < 0:
        elapsed = 0.0

    if not pass_value:
        normalized_fields = empty_fields()
    else:
        normalized_fields = {}
        source = fields if isinstance(fields, Mapping) else {}
        for name in FIELD_NAMES:
            normalized_fields[name] = _field_from_mapping(source.get(name))

    return {
        "quality": {"pass": pass_value, "issues": _issues(issues, pass_value)},
        "fields": normalized_fields,
        # Milliseconds are intentionally rounded for a stable, JSON-native
        # metric while preserving enough detail for the UI.
        "processing_ms": int(round(elapsed)),
    }

