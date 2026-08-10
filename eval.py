#!/usr/bin/env python3
"""Evaluate receipt extraction predictions against local ground-truth labels.

The evaluator intentionally has no dependency on the Streamlit interface. It
loads each labelled receipt image, calls src.pipeline.extract, and scores the
four fields described in the take-home brief.

Supported label shapes:

* labels.json (preferred), either a list of records or a mapping from an image
  filename to its field values.
* labels.csv with an image column (for example image or filename) and one
  column per field.

JSON records may also wrap values in a fields mapping. This keeps evaluation
data easy to maintain while rejecting ambiguous or unusable records with clear
warnings instead of guessing.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

try:
    from rapidfuzz import fuzz
except ImportError:  # pragma: no cover - requirements.txt pins RapidFuzz.
    fuzz = None


FIELD_NAMES: tuple[str, ...] = (
    "merchant_name",
    "transaction_date",
    "total_amount",
    "currency",
)
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"})
IMAGE_REFERENCE_KEYS: tuple[str, ...] = (
    "image",
    "image_path",
    "filename",
    "file_name",
    "file",
    "path",
)
JSON_CONTAINER_KEYS: tuple[str, ...] = ("labels", "records", "receipts", "data", "images")
ISO_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
CENT = Decimal("0.01")


class EvaluationError(Exception):
    """An input problem that should be reported without a traceback."""


@dataclass(frozen=True)
class LabelRecord:
    """One labelled image and its ground-truth field values."""

    image_reference: str
    fields: Mapping[str, Any]
    source: str


@dataclass
class FieldStats:
    """Counters used for an individual field and for the all-fields total."""

    labelled: int = 0
    correct: int = 0
    abstained: int = 0
    reviewed: int = 0
    zero_credit: int = 0
    trust_points: int = 0

    def record(self, *, correct: bool, abstained: bool, needs_review: bool) -> None:
        self.labelled += 1
        self.correct += int(correct)
        self.abstained += int(abstained)
        self.reviewed += int(needs_review)

        # The brief says abstention is free. A null/blank value therefore has
        # the same zero-credit outcome as an explicit review flag, even if a
        # malformed pipeline response forgot to set needs_review.
        if abstained or needs_review:
            self.zero_credit += 1
            return

        self.trust_points += 1 if correct else -1

    @property
    def accuracy(self) -> float | None:
        return self.correct / self.labelled if self.labelled else None

    @property
    def abstention_rate(self) -> float | None:
        return self.abstained / self.labelled if self.labelled else None

    @property
    def review_rate(self) -> float | None:
        return self.reviewed / self.labelled if self.labelled else None

    @property
    def zero_credit_rate(self) -> float | None:
        return self.zero_credit / self.labelled if self.labelled else None

    @property
    def trust_score(self) -> float | None:
        return self.trust_points / self.labelled if self.labelled else None


@dataclass
class EvaluationResult:
    """The evaluated sample counts and score aggregates."""

    label_path: Path
    images_discovered: int
    label_records_loaded: int
    images_with_labels: int
    images_processed: int
    pipeline_errors: int
    field_stats: Mapping[str, FieldStats]
    overall: FieldStats
    warnings: Sequence[str]


class Diagnostics:
    """Collect warnings so normal output remains a compact score report."""

    def __init__(self) -> None:
        self.messages: list[str] = []
        self._seen: set[str] = set()

    def warn(self, message: str) -> None:
        if message not in self._seen:
            self.messages.append(message)
            self._seen.add(message)


def _normalise_path_key(value: str) -> str:
    """Return a platform-neutral, case-insensitive path lookup key."""

    value = value.strip().replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    return value.casefold()


def _text(value: Any) -> str | None:
    """Convert useful scalar values to stripped text, rejecting containers."""

    if value is None or isinstance(value, (Mapping, list, tuple, set)):
        return None
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError:
            return None
    text = str(value).strip()
    return text or None


def _normalise_merchant(value: Any) -> str | None:
    text = _text(value)
    if text is None:
        return None
    # Whitespace is presentational here; all other character differences are
    # left for RapidFuzz's ratio so matching remains transparent.
    return re.sub(r"\s+", " ", text).casefold()


def _normalise_iso_date(value: Any) -> str | None:
    """Accept only canonical YYYY-MM-DD dates that are calendar-valid."""

    if isinstance(value, datetime):
        return None
    if isinstance(value, date):
        return value.isoformat()

    text = _text(value)
    if text is None or not ISO_DATE_PATTERN.fullmatch(text):
        return None
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError:
        return None


def _normalise_amount(value: Any) -> Decimal | None:
    """Convert an amount to cents without silently rounding a third decimal."""

    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        value = value.strip().replace(",", "")
        if not value:
            return None

    try:
        decimal_value = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None

    if not decimal_value.is_finite():
        return None
    try:
        cents_value = decimal_value.quantize(CENT)
    except (InvalidOperation, ValueError):
        return None
    # Decimal equality deliberately treats 12, 12.0 and 12.00 as the same
    # monetary value, but rejects values such as 12.001 instead of rounding.
    return cents_value if decimal_value == cents_value else None


def _normalise_currency(value: Any) -> str | None:
    """Currency matching is exact; only accidental surrounding whitespace is removed."""

    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _is_abstention(value: Any) -> bool:
    """Treat null and blank values as an honest refusal to make a prediction."""

    return value is None or (isinstance(value, str) and not value.strip())


def _merchant_similarity(left: str, right: str) -> float:
    """Return a [0, 1] ratio, using RapidFuzz whenever it is installed."""

    if fuzz is not None:
        return fuzz.ratio(left, right) / 100.0
    return SequenceMatcher(a=left, b=right).ratio()


def _is_correct(field_name: str, prediction: Any, expected: Any) -> bool:
    """Apply the field-level matching contract from section 05 of the brief."""

    if field_name == "merchant_name":
        prediction_text = _normalise_merchant(prediction)
        expected_text = _normalise_merchant(expected)
        return (
            prediction_text is not None
            and expected_text is not None
            and _merchant_similarity(prediction_text, expected_text) >= 0.85
        )

    if field_name == "transaction_date":
        expected_date = _normalise_iso_date(expected)
        return expected_date is not None and _normalise_iso_date(prediction) == expected_date

    if field_name == "total_amount":
        expected_amount = _normalise_amount(expected)
        return expected_amount is not None and _normalise_amount(prediction) == expected_amount

    if field_name == "currency":
        expected_currency = _normalise_currency(expected)
        return expected_currency is not None and _normalise_currency(prediction) == expected_currency

    raise ValueError(f"Unknown field: {field_name}")


def _valid_ground_truth(field_name: str, value: Any) -> bool:
    """Return whether a ground-truth value can be evaluated under the contract."""

    if field_name == "merchant_name":
        return _normalise_merchant(value) is not None
    if field_name == "transaction_date":
        return _normalise_iso_date(value) is not None
    if field_name == "total_amount":
        return _normalise_amount(value) is not None
    if field_name == "currency":
        return _normalise_currency(value) is not None
    return False


def _unwrap_label_value(value: Any) -> Any:
    """Allow optional {value: ...} label wrappers without accepting predictions."""

    if isinstance(value, Mapping) and "value" in value:
        return value["value"]
    return value


def _field_mapping(raw_record: Mapping[str, Any]) -> Mapping[str, Any]:
    """Find the field mapping in a JSON record."""

    for key in ("fields", "ground_truth", "groundTruth"):
        nested = raw_record.get(key)
        if isinstance(nested, Mapping):
            return nested
    return raw_record


def _image_reference(raw_record: Mapping[str, Any], fallback: str | None) -> str | None:
    """Find an image path in a record, otherwise use the enclosing map key."""

    for key in IMAGE_REFERENCE_KEYS:
        value = raw_record.get(key)
        text = _text(value)
        if text is not None:
            return text
    return fallback


def _record_from_mapping(
    raw_record: Any,
    *,
    source: str,
    fallback_image: str | None,
    diagnostics: Diagnostics,
) -> LabelRecord | None:
    if not isinstance(raw_record, Mapping):
        diagnostics.warn(f"{source}: expected an object; skipping malformed label record.")
        return None

    image_reference = _image_reference(raw_record, fallback_image)
    if image_reference is None:
        diagnostics.warn(
            f"{source}: missing image reference (expected one of {', '.join(IMAGE_REFERENCE_KEYS)}); "
            "skipping record."
        )
        return None

    fields = _field_mapping(raw_record)
    if not any(name in fields for name in FIELD_NAMES):
        diagnostics.warn(f"{source}: contains no recognised receipt fields; skipping record.")
        return None

    return LabelRecord(image_reference=image_reference, fields=fields, source=source)


def _records_from_json(payload: Any, label_path: Path, diagnostics: Diagnostics) -> list[LabelRecord]:
    """Parse common, intentionally simple JSON label layouts."""

    raw_records: Iterable[tuple[str | None, Any, str]]
    label_name = label_path.name

    if isinstance(payload, list):
        raw_records = (
            (None, raw_record, f"{label_name}[{index}]") for index, raw_record in enumerate(payload)
        )
    elif isinstance(payload, Mapping):
        container: Any | None = None
        container_name: str | None = None
        for key in JSON_CONTAINER_KEYS:
            value = payload.get(key)
            if isinstance(value, (list, Mapping)):
                container = value
                container_name = key
                break

        if container is not None:
            if isinstance(container, list):
                raw_records = (
                    (
                        None,
                        raw_record,
                        f"{label_name}.{container_name}[{index}]",
                    )
                    for index, raw_record in enumerate(container)
                )
            else:
                raw_records = (
                    (
                        str(image_reference),
                        raw_record,
                        f"{label_name}.{container_name}[{image_reference!r}]",
                    )
                    for image_reference, raw_record in container.items()
                )
        elif any(key in payload for key in IMAGE_REFERENCE_KEYS) and any(
            key in _field_mapping(payload) for key in FIELD_NAMES
        ):
            raw_records = ((None, payload, label_name),)
        else:
            # A filename-to-labels map is the compact, recommended form.
            raw_records = (
                (str(image_reference), raw_record, f"{label_name}[{image_reference!r}]")
                for image_reference, raw_record in payload.items()
            )
    else:
        raise EvaluationError(
            f"{label_path}: JSON labels must be a list of records or an object keyed by image path."
        )

    records: list[LabelRecord] = []
    for fallback_image, raw_record, source in raw_records:
        record = _record_from_mapping(
            raw_record,
            source=source,
            fallback_image=fallback_image,
            diagnostics=diagnostics,
        )
        if record is not None:
            records.append(record)
    return records


def _casefolded_row(row: Mapping[str | None, Any]) -> dict[str, Any]:
    """Map CSV header names case-insensitively while retaining value strings."""

    return {str(key).strip().casefold(): value for key, value in row.items() if key is not None}


def _records_from_csv(label_path: Path, diagnostics: Diagnostics) -> list[LabelRecord]:
    """Read CSV labels with a conventional image/path column."""

    try:
        with label_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise EvaluationError(f"{label_path}: CSV has no header row.")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise EvaluationError(f"Could not read CSV labels at {label_path}: {error}") from error

    records: list[LabelRecord] = []
    for line_number, raw_row in enumerate(rows, start=2):
        row = _casefolded_row(raw_row)
        image_reference = next(
            (_text(row.get(key)) for key in IMAGE_REFERENCE_KEYS if _text(row.get(key)) is not None),
            None,
        )
        source = f"{label_path.name}:line {line_number}"
        if image_reference is None:
            diagnostics.warn(
                f"{source}: missing image reference (expected one of {', '.join(IMAGE_REFERENCE_KEYS)}); "
                "skipping record."
            )
            continue

        fields = {field_name: row[field_name] for field_name in FIELD_NAMES if field_name in row}
        if not fields:
            diagnostics.warn(f"{source}: contains no recognised receipt fields; skipping record.")
            continue
        records.append(LabelRecord(image_reference=image_reference, fields=fields, source=source))
    return records


def _find_label_file(data_dir: Path, specified_path: Path | None = None) -> Path:
    """Choose labels.json ahead of labels.csv and generic JSON/CSV files."""

    if specified_path is not None:
        label_path = specified_path if specified_path.is_absolute() else data_dir / specified_path
        if not label_path.is_file():
            raise EvaluationError(f"Labels file does not exist: {label_path}")
        if label_path.suffix.casefold() not in {".json", ".csv"}:
            raise EvaluationError(f"Labels file must be JSON or CSV: {label_path}")
        return label_path

    files = sorted((path for path in data_dir.rglob("*") if path.is_file()), key=lambda path: str(path))
    preferred_json = [path for path in files if path.name.casefold() == "labels.json"]
    preferred_csv = [path for path in files if path.name.casefold() == "labels.csv"]
    any_json = [path for path in files if path.suffix.casefold() == ".json"]
    any_csv = [path for path in files if path.suffix.casefold() == ".csv"]

    for candidates in (preferred_json, preferred_csv, any_json, any_csv):
        if candidates:
            return candidates[0]

    raise EvaluationError(
        f"No labels file found in {data_dir}. Add labels.json (preferred) or a CSV/JSON labels file."
    )


def _load_label_records(label_path: Path, diagnostics: Diagnostics) -> list[LabelRecord]:
    """Load JSON or CSV labels, converting parsing failures to concise errors."""

    if label_path.suffix.casefold() == ".csv":
        records = _records_from_csv(label_path, diagnostics)
    else:
        try:
            with label_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise EvaluationError(f"Could not read JSON labels at {label_path}: {error}") from error
        records = _records_from_json(payload, label_path, diagnostics)

    if not records:
        raise EvaluationError(f"No usable label records found in {label_path}.")
    return records


def _discover_images(data_dir: Path) -> list[Path]:
    """Find supported image files recursively in deterministic path order."""

    return sorted(
        (
            path
            for path in data_dir.rglob("*")
            if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
        ),
        key=lambda path: str(path),
    )


def _index_images(images: Iterable[Path], data_dir: Path) -> dict[str, list[Path]]:
    """Index each image by relative path, filename, and absolute path."""

    index: dict[str, list[Path]] = defaultdict(list)
    for image_path in images:
        relative_path = image_path.relative_to(data_dir).as_posix()
        keys = {
            _normalise_path_key(relative_path),
            _normalise_path_key(image_path.name),
            _normalise_path_key(str(image_path.resolve())),
        }
        for key in keys:
            index[key].append(image_path)
    return index


def _match_image(
    record: LabelRecord,
    image_index: Mapping[str, Sequence[Path]],
    diagnostics: Diagnostics,
) -> Path | None:
    """Resolve a label's image reference while refusing ambiguous basenames."""

    matches = image_index.get(_normalise_path_key(record.image_reference), ())
    if not matches:
        diagnostics.warn(
            f"{record.source}: image {record.image_reference!r} was not found under the data directory; "
            "skipping record."
        )
        return None
    if len(matches) > 1:
        choices = ", ".join(path.as_posix() for path in matches)
        diagnostics.warn(
            f"{record.source}: image reference {record.image_reference!r} is ambiguous ({choices}); "
            "use a relative path in the labels."
        )
        return None
    return matches[0]


def _prediction_for_field(
    extraction: Any,
    field_name: str,
    *,
    image_path: Path,
    diagnostics: Diagnostics,
) -> tuple[Any, bool]:
    """Read a pipeline field safely, returning (value, needs_review)."""

    if not isinstance(extraction, Mapping):
        diagnostics.warn(f"{image_path.name}: extract() returned a non-object result; treating fields as abstained.")
        return None, True

    fields = extraction.get("fields")
    if not isinstance(fields, Mapping):
        diagnostics.warn(f"{image_path.name}: extract() result has no fields object; treating fields as abstained.")
        return None, True

    raw_field = fields.get(field_name)
    if not isinstance(raw_field, Mapping):
        diagnostics.warn(
            f"{image_path.name}: missing or malformed {field_name!r} prediction; treating it as abstained."
        )
        return None, True

    value = raw_field.get("value")
    raw_needs_review = raw_field.get("needs_review", False)
    if not isinstance(raw_needs_review, bool):
        diagnostics.warn(
            f"{image_path.name}: {field_name!r}.needs_review is not a boolean; treating it as not flagged."
        )
        raw_needs_review = False
    return value, raw_needs_review


def evaluate(
    data_dir: str | Path,
    *,
    labels_path: str | Path | None = None,
    extractor: Callable[[str], Mapping[str, Any]] | None = None,
) -> EvaluationResult:
    """Evaluate all labelled images in data_dir.

    extractor is injectable for tests. In normal use the required
    src.pipeline.extract implementation is imported lazily, so a labels
    parsing error never fails because an OCR dependency is unavailable.
    """

    resolved_data_dir = Path(data_dir).expanduser().resolve()
    if not resolved_data_dir.is_dir():
        raise EvaluationError(f"Data directory does not exist or is not a directory: {resolved_data_dir}")

    diagnostics = Diagnostics()
    selected_labels_path = _find_label_file(
        resolved_data_dir,
        Path(labels_path).expanduser() if labels_path is not None else None,
    )
    records = _load_label_records(selected_labels_path, diagnostics)
    images = _discover_images(resolved_data_dir)
    if not images:
        raise EvaluationError(
            f"No receipt images found in {resolved_data_dir}. Supported extensions: "
            f"{', '.join(sorted(IMAGE_EXTENSIONS))}."
        )

    image_index = _index_images(images, resolved_data_dir)
    records_by_image: dict[Path, LabelRecord] = {}
    for record in records:
        image_path = _match_image(record, image_index, diagnostics)
        if image_path is None:
            continue
        if image_path in records_by_image:
            diagnostics.warn(
                f"{record.source}: duplicate labels for {image_path.name}; keeping "
                f"{records_by_image[image_path].source} and skipping this record."
            )
            continue
        records_by_image[image_path] = record

    if not records_by_image:
        raise EvaluationError(
            "No label records matched discovered images. Check image filenames and paths in the labels file."
        )

    for image_path in images:
        if image_path not in records_by_image:
            relative_path = image_path.relative_to(resolved_data_dir).as_posix()
            diagnostics.warn(f"No label record for image {relative_path!r}; it was not evaluated.")

    if extractor is None:
        try:
            from src.pipeline import extract as pipeline_extract
        except Exception as error:  # noqa: BLE001 - dependency failures are actionable CLI errors.
            raise EvaluationError(
                "Could not import src.pipeline.extract. Ensure the pipeline and its dependencies are installed: "
                f"{error}"
            ) from error
        extractor = pipeline_extract

    field_stats = {field_name: FieldStats() for field_name in FIELD_NAMES}
    overall = FieldStats()
    images_processed = 0
    pipeline_errors = 0

    for image_path in sorted(records_by_image, key=lambda path: str(path)):
        record = records_by_image[image_path]
        try:
            extraction = extractor(str(image_path))
            images_processed += 1
        except Exception as error:  # noqa: BLE001 - report an individual bad image, continue evaluation.
            pipeline_errors += 1
            diagnostics.warn(
                f"{image_path.name}: extract() raised {type(error).__name__}: {error}; image was not scored."
            )
            continue

        for field_name in FIELD_NAMES:
            if field_name not in record.fields:
                diagnostics.warn(f"{record.source}: missing ground-truth {field_name!r}; field was not scored.")
                continue

            expected = _unwrap_label_value(record.fields[field_name])
            if not _valid_ground_truth(field_name, expected):
                diagnostics.warn(
                    f"{record.source}: invalid ground-truth {field_name!r} value "
                    f"{expected!r}; field was not scored."
                )
                continue

            prediction, needs_review = _prediction_for_field(
                extraction,
                field_name,
                image_path=image_path,
                diagnostics=diagnostics,
            )
            abstained = _is_abstention(prediction)
            correct = not abstained and _is_correct(field_name, prediction, expected)

            field_stats[field_name].record(
                correct=correct,
                abstained=abstained,
                needs_review=needs_review,
            )
            overall.record(
                correct=correct,
                abstained=abstained,
                needs_review=needs_review,
            )

    if overall.labelled == 0:
        raise EvaluationError(
            "No valid labelled fields were scored. Check required field names and label value formats."
        )

    return EvaluationResult(
        label_path=selected_labels_path,
        images_discovered=len(images),
        label_records_loaded=len(records),
        images_with_labels=len(records_by_image),
        images_processed=images_processed,
        pipeline_errors=pipeline_errors,
        field_stats=field_stats,
        overall=overall,
        warnings=tuple(diagnostics.messages),
    )


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:6.2f}%"


def _trust(points: int, score: float | None) -> str:
    return "n/a" if score is None else f"{score:+.3f} ({points:+d})"


def print_report(result: EvaluationResult) -> None:
    """Print a human-readable, stable report suitable for REPORT.md copy/paste."""

    print("Receipt extraction evaluation")
    print(f"Labels: {result.label_path}")
    print(
        "Images: "
        f"{result.images_discovered} discovered, "
        f"{result.images_with_labels} matched to labels, "
        f"{result.images_processed} processed"
    )
    if result.pipeline_errors:
        print(f"Pipeline errors: {result.pipeline_errors} (those images were not scored)")
    if fuzz is None:
        print("Matcher: RapidFuzz unavailable; using a compatibility fallback. Install requirements.txt for final metrics.")
    else:
        print("Matcher: RapidFuzz ratio (merchant threshold >= 0.85)")
    print()
    print(
        f"{'Field':<20}"
        f"{'Accuracy':>11}  "
        f"{'Abstain':>10}  "
        f"{'Review':>10}  "
        f"{'Zero credit':>12}  "
        f"{'Trust':>14}  "
        f"{'Scored':>8}"
    )
    print("-" * 98)
    for field_name in FIELD_NAMES:
        stats = result.field_stats[field_name]
        print(
            f"{field_name:<20}"
            f"{_percent(stats.accuracy):>11}  "
            f"{_percent(stats.abstention_rate):>10}  "
            f"{_percent(stats.review_rate):>10}  "
            f"{_percent(stats.zero_credit_rate):>12}  "
            f"{_trust(stats.trust_points, stats.trust_score):>14}  "
            f"{stats.labelled:>8}"
        )
    print("-" * 98)
    overall = result.overall
    print(
        f"{'ALL FIELDS':<20}"
        f"{_percent(overall.accuracy):>11}  "
        f"{_percent(overall.abstention_rate):>10}  "
        f"{_percent(overall.review_rate):>10}  "
        f"{_percent(overall.zero_credit_rate):>12}  "
        f"{_trust(overall.trust_points, overall.trust_score):>14}  "
        f"{overall.labelled:>8}"
    )
    print()
    print(
        "Trust scoring: +1 correct and not flagged, 0 reviewed/abstained, "
        "-1 wrong and not flagged."
    )
    print(
        f"Trust score: {overall.trust_score:+.3f} "
        f"({overall.trust_points:+d} points across {overall.labelled} labelled fields)"
    )

    if result.warnings:
        print(f"\nWarnings ({len(result.warnings)}):", file=sys.stderr)
        for warning in result.warnings:
            print(f"  - {warning}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate src.pipeline.extract against receipt labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        required=True,
        metavar="DIR",
        help="Directory containing receipt images and labels.json or labels.csv.",
    )
    parser.add_argument(
        "--labels",
        metavar="FILE",
        help="Optional labels file path, relative to --data unless absolute.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = evaluate(args.data, labels_path=args.labels)
    except EvaluationError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    print_report(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
