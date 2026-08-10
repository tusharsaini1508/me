"""Deterministic, receipt-aware parsing of local OCR output.

This module has no OpenCV or Tesseract dependency.  It receives typed OCR
tokens/lines and returns conservative field candidates, which makes the
business rules straightforward to unit test separately from image handling.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Iterable, Mapping, Sequence

from .schema import FieldDict, make_field


_WHITESPACE = re.compile(r"\s+")
_DATE_NUMERIC = re.compile(
    r"(?<!\d)(?P<a>\d{1,4})\s*[-/.]\s*(?P<b>\d{1,2})\s*[-/.]\s*(?P<c>\d{2,4})(?!\d)"
)
_DATE_MONTH_FIRST = re.compile(
    r"(?<![A-Za-z])(?P<month>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
    r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?)\.?\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?[,]?\s+(?P<year>\d{2,4})(?!\d)",
    re.IGNORECASE,
)
_DATE_DAY_FIRST = re.compile(
    r"(?<!\d)(?P<day>\d{1,2})(?:st|nd|rd|th)?\s+"
    r"(?P<month>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?[,]?\s+"
    r"(?P<year>\d{2,4})(?!\d)",
    re.IGNORECASE,
)
_AMOUNT = re.compile(
    r"(?<![A-Za-z0-9])(?P<symbol>₹|Rs\.?|INR|\$|US\$|USD|€|EUR|£|GBP|AED|CAD|AUD|SGD)?\s*"
    r"(?P<number>\d{1,3}(?:[ ,.\u00a0]\d{3})*(?:[,.]\d{2})|\d+(?:[,.]\d{2})|\d+)(?![A-Za-z0-9])",
    re.IGNORECASE,
)

_MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

_DATE_LABEL = re.compile(
    r"\b(?:date|dated|invoice[_\s-]*date|bill[_\s-]*date|"
    r"transaction(?:[_\s-]*date)?|txn(?:[_\s-]*date)?)\b",
    re.IGNORECASE,
)
_TOTAL_STRONG = re.compile(
    r"\b(grand[_\s-]*total|amount[_\s-]*(?:due|payable)|net[_\s-]*amount|"
    r"total[_\s-]*(?:amount|due|payable))\b",
    re.IGNORECASE,
)
_TOTAL_GENERIC = re.compile(r"\btotal\b|\btotal[_\s-]*amount\b", re.IGNORECASE)
_TOTAL_NEGATIVE = re.compile(
    r"\b(sub[_\s-]*total|subtotal|tax|gst|vat|cgst|sgst|discount|change|cash|tender|"
    r"round(?:ing|[_\s-]*off)?|balance)\b",
    re.IGNORECASE,
)
_GENERIC_MERCHANT = re.compile(
    r"\b(receipt|tax\s*invoice|invoice|cash\s*memo|bill|customer\s*copy|duplicate|"
    r"thank\s*you|welcome|sales\s*slip|terminal\s*copy|payment\s*summary|"
    r"transaction\s+has|successfully\s+processed)\b",
    re.IGNORECASE,
)
_MERCHANT_NOISE = re.compile(
    r"\b(tel(?:ephone)?|phone|mobile|www\.?|http|gstin|tin\b|pan\b|invoice\s*(?:no|#)|"
    r"order\s*(?:no|#)|table\s*(?:no|#)|cashier|server)\b|[@#]",
    re.IGNORECASE,
)
_MERCHANT_LABEL = re.compile(
    r"\bmerchant(?:[_\s-]*name)?\b\s*(?:[:=|]\s*)?(?P<value>.+)$",
    re.IGNORECASE,
)
_STRUCTURED_LABEL = re.compile(
    r"^\s*(?:merchant\s+name|store\s+name|shop\s+name|(?P<label_word>merchant|store|shop|name|date|amount|total|price|currency|code))\s*[:=|.]?\s*(?P<value>.+)$",
    re.IGNORECASE,
)
_LABEL_VALUE_PREFIX = re.compile(r"^[\s:;=|#\-]+")
_LABEL_VALUE_SUFFIX = re.compile(r"\s*(?:[|>]+|\(\s*[A-Za-z0-9]{0,3}[>)]?)\s*$")


@dataclass(frozen=True)
class OCRToken:
    """One OCR word and its geometry in the rectified image."""

    text: str
    confidence: float = 0.0
    left: int = 0
    top: int = 0
    width: int = 0
    height: int = 0
    block_num: int = 0
    par_num: int = 0
    line_num: int = 0

    @property
    def confidence_fraction(self) -> float:
        try:
            value = float(self.confidence) / 100.0
        except (TypeError, ValueError):
            return 0.0
        return min(1.0, max(0.0, value)) if math.isfinite(value) else 0.0


@dataclass(frozen=True)
class OCRLine:
    """A visual OCR line, retaining the words used to derive confidence."""

    text: str
    tokens: tuple[OCRToken, ...] = ()
    index: int = 0
    top: int = 0
    left: int = 0
    width: int = 0
    height: int = 0

    @property
    def normalized_text(self) -> str:
        return normalize_whitespace(self.text)

    @property
    def confidence(self) -> float:
        if not self.tokens:
            # Hand-authored unit-test lines are still useful.  Their evidence
            # should not become a high-confidence production result.
            return 0.55
        usable = [token.confidence_fraction for token in self.tokens if token.text.strip()]
        return sum(usable) / len(usable) if usable else 0.0


@dataclass(frozen=True)
class ParsedField:
    """A parser candidate before the public schema is materialized."""

    value: str | float | None
    confidence: float = 0.0
    needs_review: bool = True
    evidence: str = ""

    def as_dict(self) -> FieldDict:
        return make_field(self.value, self.confidence, self.needs_review)


def normalize_whitespace(value: str) -> str:
    """Collapse OCR spacing without changing printed spelling/case."""

    return _WHITESPACE.sub(" ", str(value)).strip()


def _evidence_token(value: str) -> str:
    """Normalize one OCR token for inexpensive evidence matching."""

    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _value_token_confidence(line: OCRLine, value: str) -> float | None:
    """Return confidence for tokens that actually form a selected field value.

    A full OCR line often includes a row number, label, and decorative noise.
    Scoring a labelled amount/name from its own tokens prevents those unrelated
    words from making a bad value look trustworthy.
    """

    if not line.tokens:
        return None
    expected = {_evidence_token(part) for part in value.split()}
    expected.discard("")
    if not expected:
        return None
    matched = [
        token.confidence_fraction
        for token in line.tokens
        if _evidence_token(token.text) in expected
    ]
    return sum(matched) / len(matched) if matched else None


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if math.isfinite(numeric) else default


def tokens_from_ocr_data(data: Mapping[str, Sequence[object]]) -> list[OCRToken]:
    """Convert a ``pytesseract.image_to_data`` dictionary into typed tokens."""

    raw_text = data.get("text", ())
    if isinstance(raw_text, str) or not isinstance(raw_text, Sequence):
        return []

    def at(name: str, index: int, default: object = 0) -> object:
        values = data.get(name, ())
        if isinstance(values, str) or not isinstance(values, Sequence) or index >= len(values):
            return default
        return values[index]

    tokens: list[OCRToken] = []
    for index, raw in enumerate(raw_text):
        text = normalize_whitespace(str(raw))
        if not text:
            continue
        tokens.append(
            OCRToken(
                text=text,
                confidence=_safe_float(at("conf", index, -1.0), -1.0),
                left=_safe_int(at("left", index)),
                top=_safe_int(at("top", index)),
                width=_safe_int(at("width", index)),
                height=_safe_int(at("height", index)),
                block_num=_safe_int(at("block_num", index)),
                par_num=_safe_int(at("par_num", index)),
                line_num=_safe_int(at("line_num", index)),
            )
        )
    return tokens


def lines_from_tokens(tokens: Iterable[OCRToken]) -> list[OCRLine]:
    """Group OCR words by Tesseract line identifiers in visual order."""

    grouped: dict[tuple[int, int, int], list[OCRToken]] = defaultdict(list)
    for ordinal, token in enumerate(tokens):
        if not token.text.strip():
            continue
        # Tesseract normally provides positive identifiers.  Geometry is a
        # deterministic fallback for hand-built or partial OCR data.
        key = (token.block_num, token.par_num, token.line_num)
        if key == (0, 0, 0):
            key = (0, token.top, ordinal if token.top == 0 else 0)
        grouped[key].append(token)

    ordered_groups = sorted(
        grouped.values(),
        key=lambda words: (min(word.top for word in words), min(word.left for word in words)),
    )
    lines: list[OCRLine] = []
    for index, words in enumerate(ordered_groups):
        ordered_words = tuple(sorted(words, key=lambda word: (word.left, word.top)))
        left = min(word.left for word in ordered_words)
        top = min(word.top for word in ordered_words)
        right = max(word.left + max(0, word.width) for word in ordered_words)
        bottom = max(word.top + max(0, word.height) for word in ordered_words)
        lines.append(
            OCRLine(
                text=" ".join(word.text for word in ordered_words),
                tokens=ordered_words,
                index=index,
                top=top,
                left=left,
                width=max(0, right - left),
                height=max(0, bottom - top),
            )
        )
    return lines


def lines_from_ocr_data(data: Mapping[str, Sequence[object]]) -> list[OCRLine]:
    """Convenience bridge used by the pipeline after local OCR completes."""

    return lines_from_tokens(tokens_from_ocr_data(data))


def _year(value: int) -> int:
    """Map two-digit receipt years conservatively to the 2000s."""

    if value < 100:
        return 2000 + value
    return value


def _format_date(year: int, month: int, day: int) -> str | None:
    if not 2000 <= year <= 2100:
        return None
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def normalize_date(value: str) -> str | None:
    """Normalize a clear date to ISO 8601, rejecting ambiguous numeric dates.

    ``03/04/2026`` is not silently interpreted as either US or day-first
    ordering.  The ranking function can retain it as a reviewed candidate,
    but callers asking for a canonical standalone value get ``None``.
    """

    text = normalize_whitespace(value)
    for pattern, day_first in ((_DATE_DAY_FIRST, True), (_DATE_MONTH_FIRST, False)):
        match = pattern.search(text)
        if match:
            month_name = match.group("month").casefold().rstrip(".")
            month = _MONTHS.get(month_name)
            if month is None:
                return None
            return _format_date(
                _year(int(match.group("year"))),
                month,
                int(match.group("day")),
            )

    match = _DATE_NUMERIC.search(text)
    if not match:
        return None
    a, b, c = (int(match.group(name)) for name in ("a", "b", "c"))
    if len(match.group("a")) == 4:
        return _format_date(a, b, c)
    year = _year(c)
    if a > 12 and b <= 12:
        return _format_date(year, b, a)
    if b > 12 and a <= 12:
        return _format_date(year, a, b)
    return None


def _date_match_candidates(text: str) -> list[tuple[str, bool]]:
    """Return ``(iso_date, ambiguous)`` values found in an OCR line."""

    candidates: list[tuple[str, bool]] = []
    for pattern, day_first in ((_DATE_DAY_FIRST, True), (_DATE_MONTH_FIRST, False)):
        for match in pattern.finditer(text):
            month = _MONTHS.get(match.group("month").casefold().rstrip("."))
            if month is None:
                continue
            iso = _format_date(_year(int(match.group("year"))), month, int(match.group("day")))
            if iso:
                candidates.append((iso, False))

    for match in _DATE_NUMERIC.finditer(text):
        a, b, c = (int(match.group(name)) for name in ("a", "b", "c"))
        first = match.group("a")
        if len(first) == 4:
            iso = _format_date(a, b, c)
            if iso:
                candidates.append((iso, False))
            continue
        year = _year(c)
        if a > 12 and b <= 12:
            iso = _format_date(year, b, a)
            if iso:
                candidates.append((iso, False))
        elif b > 12 and a <= 12:
            iso = _format_date(year, a, b)
            if iso:
                candidates.append((iso, False))
        else:
            # Keep a day-first provisional result only as reviewed evidence;
            # India is a likely target locale, but this must not look certain.
            iso = _format_date(year, b, a)
            if iso:
                candidates.append((iso, True))
    return candidates


def parse_date(lines: Sequence[OCRLine]) -> ParsedField:
    structured_candidates: list[tuple[float, str, bool, str]] = []
    candidates: list[tuple[float, str, bool, str]] = []
    
    for position, line in enumerate(lines):
        text = line.normalized_text
        
        structured_value = _extract_structured_field(text, "date")
        if structured_value is not None:
            try:
                iso = normalize_date(structured_value)
                if iso:
                    structured_score = min(0.98, 0.85 + 0.13 * line.confidence)
                    structured_candidates.append((structured_score, iso, False, text))
                    continue
            except Exception:
                pass
        
        for iso, ambiguous in _date_match_candidates(text):
            label_bonus = 0.20 if _DATE_LABEL.search(text) else 0.0
            explicit_bonus = 0.12 if not ambiguous else -0.12
            position_bonus = 0.05 if position < max(3, len(lines) // 3) else 0.0
            score = min(0.98, 0.38 + 0.38 * line.confidence + label_bonus + explicit_bonus + position_bonus)
            candidates.append((score, iso, ambiguous, text))
    
    if structured_candidates:
        structured_candidates.sort(key=lambda item: (-item[0], item[1], item[3]))
        score, value, ambiguous, evidence = structured_candidates[0]
        
        competing = [item for item in candidates if item[1] != value and score - item[0] < 0.08]
        needs_review = ambiguous or bool(competing) or score < 0.85
        return ParsedField(value, score, needs_review, evidence)
    
    if not candidates:
        return ParsedField(None)

    candidates.sort(key=lambda item: (-item[0], item[1], item[3]))
    score, value, ambiguous, evidence = candidates[0]
    competing = [item for item in candidates[1:] if item[1] != value and score - item[0] < 0.08]
    needs_review = ambiguous or bool(competing) or score < 0.80
    return ParsedField(value, score, needs_review, evidence)


def _decimal_amount(raw_value: str, *, allow_integer: bool = False) -> float | None:
    """Parse a receipt amount to a two-decimal native float without guessing."""

    compact = raw_value.replace("\u00a0", "").replace(" ", "")
    if not compact or not re.fullmatch(r"\d[\d,.]*", compact):
        return None
    decimal_separator: str | None = None
    if "." in compact and "," in compact:
        decimal_separator = "." if compact.rfind(".") > compact.rfind(",") else ","
    elif "." in compact:
        suffix = compact.rsplit(".", 1)[1]
        if len(suffix) == 2:
            decimal_separator = "."
    elif "," in compact:
        suffix = compact.rsplit(",", 1)[1]
        if len(suffix) == 2:
            decimal_separator = ","

    if decimal_separator:
        whole, fraction = compact.rsplit(decimal_separator, 1)
        normalized = whole.replace(".", "").replace(",", "") + "." + fraction
    else:
        if not allow_integer or "." in compact or "," in compact:
            return None
        normalized = compact

    try:
        amount = Decimal(normalized)
    except InvalidOperation:
        return None
    if not amount.is_finite() or amount < 0 or amount > Decimal("10000000"):
        return None
    rounded = amount.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    # Do not turn OCR values such as 12.345 into made-up cents.
    if amount != rounded:
        return None
    return float(rounded)


def normalize_amount(value: str) -> float | None:
    """Parse a clearly formatted two-decimal amount from arbitrary text."""

    for match in _AMOUNT.finditer(normalize_whitespace(value)):
        amount = _decimal_amount(match.group("number"))
        if amount is not None:
            return amount
    return None


def _amount_candidates(lines: Sequence[OCRLine]) -> list[tuple[float, float, str, str]]:
    candidates: list[tuple[float, float, str, str]] = []
    total_lines = max(1, len(lines))
    for position, line in enumerate(lines):
        text = line.normalized_text
        lower = text.casefold()
        strong_match = _TOTAL_STRONG.search(lower)
        strong = strong_match is not None
        generic = bool(_TOTAL_GENERIC.search(lower))
        negative = bool(_TOTAL_NEGATIVE.search(lower))
        for match in _AMOUNT.finditer(text):
            raw_amount = match.group("number")
            # Digit-only row numbers often appear before a labelled total in
            # digital receipts (for example ``2 total_amount 1,250.00``).
            # They are not monetary values and should not beat the actual
            # amount merely because their OCR confidence is high.
            if (
                strong_match is not None
                and match.start("number") < strong_match.start()
                and re.fullmatch(r"\d{1,3}", raw_amount)
            ):
                continue
            amount = _decimal_amount(raw_amount, allow_integer=strong or generic)
            if amount is None:
                continue
            amount_confidence = _value_token_confidence(line, raw_amount)
            ocr_confidence = line.confidence if amount_confidence is None else amount_confidence
            if strong:
                semantic = 0.92
            elif generic and not negative:
                semantic = 0.76
            elif negative:
                semantic = 0.12
            else:
                semantic = 0.28
            lower_page_bonus = 0.10 * (position / max(1, total_lines - 1))
            symbol_bonus = 0.04 if match.group("symbol") else 0.0
            score = min(0.98, 0.20 + 0.52 * semantic + 0.24 * ocr_confidence + lower_page_bonus + symbol_bonus)
            candidates.append((score, amount, text, "strong" if strong else "generic" if generic else "other"))
    return candidates


def parse_total(lines: Sequence[OCRLine]) -> ParsedField:
    structured_candidates: list[tuple[float, float, str, str]] = []
    candidates = _amount_candidates(lines)
    
    for line in lines:
        text = line.normalized_text
        structured_value = _extract_structured_field(text, "amount")
        if structured_value is not None:
            amount = normalize_amount(structured_value)
            if amount is not None:
                structured_score = min(0.98, 0.88 + 0.10 * line.confidence)
                structured_candidates.append((structured_score, amount, text, "structured"))
    
    if structured_candidates:
        structured_candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
        score, value, evidence, strength = structured_candidates[0]
        
        close_different = [
            candidate
            for candidate in candidates
            if candidate[1] != value and score - candidate[0] < 0.07
        ]
        needs_review = bool(close_different) or score < 0.85
        return ParsedField(value, score, needs_review, evidence)
    
    if not candidates:
        return ParsedField(None)
    candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
    score, value, evidence, strength = candidates[0]
    close_different = [
        candidate
        for candidate in candidates[1:]
        if candidate[1] != value and score - candidate[0] < 0.07
    ]
    # An unlabeled amount may still be useful to a human reviewer, but must
    # never look like a trusted total.
    # A clear, well-recognized ``TOTAL`` line is valid evidence in its own
    # right.  Generic merely means it did not use a longer "grand total" or
    # "amount due" phrase; ambiguous/lower-confidence candidates stay reviewed.
    needs_review = strength == "other" or bool(close_different) or score < 0.82
    return ParsedField(value, score, needs_review, evidence)


_CURRENCY_EVIDENCE: tuple[tuple[re.Pattern[str], str, bool], ...] = (
    (re.compile(r"\bINR\b|₹|\bRS\.?\b|\bRUPEES?\b", re.IGNORECASE), "INR", True),
    (re.compile(r"\bUSD\b|\bUS\$|\bUS\s+DOLLARS?\b", re.IGNORECASE), "USD", True),
    (re.compile(r"\bEUR\b|€|\bEUROS?\b", re.IGNORECASE), "EUR", True),
    (re.compile(r"\bGBP\b|£|\bPOUNDS?\b", re.IGNORECASE), "GBP", True),
    (re.compile(r"\bAED\b|\bDIRHAMS?\b", re.IGNORECASE), "AED", True),
    (re.compile(r"\bCAD\b|\bCANADIAN\s+DOLLARS?\b", re.IGNORECASE), "CAD", True),
    (re.compile(r"\bAUD\b|\bAUSTRALIAN\s+DOLLARS?\b", re.IGNORECASE), "AUD", True),
    (re.compile(r"\bSGD\b|\bSINGAPORE\s+DOLLARS?\b", re.IGNORECASE), "SGD", True),
    # A bare dollar sign is not enough to distinguish USD, CAD, AUD, etc.
    (re.compile(r"\$"), "USD", False),
)


def parse_currency(lines: Sequence[OCRLine]) -> ParsedField:
    scores: dict[str, float] = defaultdict(float)
    evidence: dict[str, str] = {}
    explicit: dict[str, bool] = defaultdict(bool)
    
    for line in lines:
        text = line.normalized_text
        
        structured_value = _extract_structured_field(text, "currency")
        if structured_value is not None:
            structured_upper = structured_value.upper()
            scores[structured_upper] += 0.95
            explicit[structured_upper] = True
            evidence.setdefault(structured_upper, text)
            continue
        
        for pattern, currency, unambiguous in _CURRENCY_EVIDENCE:
            if not pattern.search(text):
                continue
            weight = 0.36 + 0.36 * line.confidence + (0.20 if unambiguous else 0.0)
            scores[currency] += weight
            explicit[currency] = explicit[currency] or unambiguous
            evidence.setdefault(currency, text)
    if not scores:
        return ParsedField(None)

    ranked = sorted(scores, key=lambda code: (-scores[code], code))
    winner = ranked[0]
    raw_score = scores[winner]
    runner_up = scores[ranked[1]] if len(ranked) > 1 else 0.0
    confidence = min(0.97, raw_score / max(1.0, raw_score + runner_up * 0.55))
    needs_review = not explicit[winner] or runner_up > raw_score * 0.7 or confidence < 0.80
    return ParsedField(winner, confidence, needs_review, evidence[winner])


def _merchant_shape_score(text: str) -> float:
    letters = sum(character.isalpha() for character in text)
    digits = sum(character.isdigit() for character in text)
    length = len(text)
    if letters < 3 or length > 64:
        return 0.0
    alpha_ratio = letters / max(1, letters + digits)
    length_score = 1.0 if 4 <= length <= 40 else 0.65
    return 0.65 * alpha_ratio + 0.35 * length_score


def _merchant_label_value(text: str) -> str | None:
    """Extract the value from explicit merchant-name rows when OCR preserves it."""

    match = _MERCHANT_LABEL.search(text)
    if match is None:
        return None
    value = _LABEL_VALUE_PREFIX.sub("", match.group("value"))
    value = _LABEL_VALUE_SUFFIX.sub("", value)
    value = normalize_whitespace(value)
    if not value or _GENERIC_MERCHANT.search(value) or _MERCHANT_NOISE.search(value):
        return None
    if _DATE_NUMERIC.search(value) or _DATE_DAY_FIRST.search(value) or _DATE_MONTH_FIRST.search(value):
        return None
    if _AMOUNT.search(value) or _merchant_shape_score(value) < 0.42:
        return None
    return value


def _extract_structured_field(text: str, field_type: str) -> str | None:
    """Extract value from a structured labeled field (merchant: value, date: value, etc.)."""

    match = _STRUCTURED_LABEL.match(text)
    if not match:
        return None
    
    label = match.group("label_word")
    if label is None:
        label = "merchant" if "merchant" in text.casefold() else "store" if "store" in text.casefold() else "shop" if "shop" in text.casefold() else "name"
    
    value = _LABEL_VALUE_PREFIX.sub("", match.group("value"))
    value = _LABEL_VALUE_SUFFIX.sub("", value)
    value = normalize_whitespace(value)
    
    if not value:
        return None
    
    field_type_lower = field_type.casefold()
    
    if field_type_lower == "merchant" and label in ("merchant", "store", "shop", "name"):
        if _GENERIC_MERCHANT.search(value) or _MERCHANT_NOISE.search(value):
            return None
        if _DATE_NUMERIC.search(value) or _AMOUNT.search(value):
            return None
        return value if _merchant_shape_score(value) >= 0.42 else None
    
    if field_type_lower == "date" and label in ("date",):
        if _DATE_NUMERIC.search(value) or _DATE_DAY_FIRST.search(value) or _DATE_MONTH_FIRST.search(value):
            return value
    
    if field_type_lower == "amount" and label in ("amount", "total", "price"):
        if _AMOUNT.search(value):
            return value
    
    if field_type_lower == "currency" and label in ("currency", "code"):
        return value
    
    return None


def parse_merchant(lines: Sequence[OCRLine]) -> ParsedField:
    """Extract merchant name from OCR lines, checking structured labels first."""

    labelled_candidates: list[tuple[float, str, str]] = []
    structured_candidates: list[tuple[float, str, str]] = []
    
    for line in lines:
        text = line.normalized_text
        structured_value = _extract_structured_field(text, "merchant")
        if structured_value is not None:
            value_confidence = _value_token_confidence(line, structured_value)
            ocr_confidence = line.confidence if value_confidence is None else value_confidence
            structured_score = min(0.99, 0.85 + 0.14 * ocr_confidence)
            structured_candidates.append((structured_score, structured_value, text))
            continue

        value = _merchant_label_value(text)
        if value is None:
            continue
        value_confidence = _value_token_confidence(line, value)
        ocr_confidence = line.confidence if value_confidence is None else value_confidence
        score = min(0.99, 0.60 + 0.36 * ocr_confidence + 0.04 * _merchant_shape_score(value))
        labelled_candidates.append((score, value, text))
    
    if structured_candidates:
        structured_candidates.sort(key=lambda item: (-item[0], item[1].casefold()))
        score, value, evidence = structured_candidates[0]
        
        competitor = any(
            candidate[1].casefold() != value.casefold() and score - candidate[0] < 0.08
            for candidate in labelled_candidates
        ) or any(
            candidate[1].casefold() != value.casefold() and score - candidate[0] < 0.08
            for candidate in structured_candidates[1:]
        )
        return ParsedField(value, score, competitor or score < 0.90, evidence)
    
    if labelled_candidates:
        labelled_candidates.sort(key=lambda item: (-item[0], item[1].casefold()))
        score, value, evidence = labelled_candidates[0]
        competitor = any(
            candidate[1].casefold() != value.casefold() and score - candidate[0] < 0.08
            for candidate in labelled_candidates[1:]
        )
        return ParsedField(value, score, competitor or score < 0.78, evidence)

    candidates: list[tuple[float, str, str]] = []
    for position, line in enumerate(lines[:8]):
        text = line.normalized_text
        if not text or _GENERIC_MERCHANT.search(text) or _MERCHANT_NOISE.search(text):
            continue
        if _DATE_NUMERIC.search(text) or _DATE_DAY_FIRST.search(text) or _DATE_MONTH_FIRST.search(text):
            continue
        if _AMOUNT.search(text):
            continue
        shape = _merchant_shape_score(text)
        if shape <= 0:
            continue
        position_score = max(0.25, 1.0 - position * 0.13)
        score = min(0.96, 0.18 + 0.38 * position_score + 0.31 * line.confidence + 0.20 * shape)
        candidates.append((score, text, line.normalized_text))
    if not candidates:
        return ParsedField(None)

    candidates.sort(key=lambda item: (-item[0], item[1].casefold()))
    score, value, evidence = candidates[0]
    competitor = any(
        candidate[1].casefold() != value.casefold() and score - candidate[0] < 0.06
        for candidate in candidates[1:]
    )
    return ParsedField(value, score, competitor or score < 0.80, evidence)


def parse_receipt_fields(lines: Sequence[OCRLine]) -> dict[str, FieldDict]:
    """Return all public fields from OCR lines using deterministic rankings."""

    normalized_lines = [
        line if isinstance(line, OCRLine) else OCRLine(str(line))
        for line in lines
        if normalize_whitespace(line.text if isinstance(line, OCRLine) else str(line))
    ]
    merchant = parse_merchant(normalized_lines)
    transaction_date = parse_date(normalized_lines)
    total = parse_total(normalized_lines)
    currency = parse_currency(normalized_lines)
    return {
        "merchant_name": merchant.as_dict(),
        "transaction_date": transaction_date.as_dict(),
        "total_amount": total.as_dict(),
        "currency": currency.as_dict(),
    }


# British spellings remain convenient for callers reading the project brief.
normalise_date = normalize_date
normalise_amount = normalize_amount
