"""Deterministic unit tests for receipt text parsing helpers.

The examples use hand-authored OCR lines, so parser regressions can be diagnosed
without invoking Tesseract or loading an image.  The test module skips only
until ``src.parsing`` is introduced; a present-but-incomplete module fails
normally and exposes the missing API.
"""

from __future__ import annotations

import math
import unittest
from collections.abc import Mapping
from typing import Any

try:
    from src import parsing
except ImportError:  # The parser is introduced independently of this scaffold.
    parsing = None  # type: ignore[assignment]


EXPECTED_FIELD_NAMES = (
    "merchant_name",
    "transaction_date",
    "total_amount",
    "currency",
)


@unittest.skipIf(parsing is None, "src.parsing has not been added yet")
class ReceiptParsingTests(unittest.TestCase):
    def test_normalize_date_returns_canonical_calendar_valid_iso_date(self) -> None:
        self.assertEqual(parsing.normalize_date("14/03/2026"), "2026-03-14")
        self.assertEqual(parsing.normalize_date("2026-03-14"), "2026-03-14")
        self.assertIsNone(parsing.normalize_date("31/02/2026"))
        self.assertIsNone(parsing.normalize_date("not a date"))

    def test_parser_extracts_clear_fields_from_ordered_ocr_lines(self) -> None:
        lines = (
            self._line(0, "CAFE NINE", top=28),
            self._line(1, "14/03/2026 11:42", top=90),
            self._line(2, "Masala chai       INR 80.00", top=190),
            self._line(3, "Lunch bowl        INR 400.00", top=238),
            self._line(4, "TOTAL             INR 480.00", top=340),
        )

        fields = parsing.parse_receipt_fields(lines)

        self.assertEqual(tuple(fields), EXPECTED_FIELD_NAMES)
        self.assertEqual(str(fields["merchant_name"]["value"]).casefold(), "cafe nine")
        self.assertEqual(fields["transaction_date"]["value"], "2026-03-14")
        self.assertEqual(fields["total_amount"]["value"], 480.0)
        self.assertEqual(fields["currency"]["value"], "INR")
        for name in EXPECTED_FIELD_NAMES:
            self._assert_field_contract(fields[name])

    def test_parser_abstains_safely_when_there_is_no_total_evidence(self) -> None:
        lines = (
            self._line(0, "CAFE NINE", top=28),
            self._line(1, "14/03/2026 11:42", top=90),
            self._line(2, "Thank you for visiting", top=190),
        )

        fields = parsing.parse_receipt_fields(lines)

        self._assert_field_contract(fields["total_amount"])
        self.assertIsNone(fields["total_amount"]["value"])
        self.assertIs(fields["total_amount"]["needs_review"], True)
        self._assert_field_contract(fields["currency"])
        self.assertIsNone(fields["currency"]["value"])
        self.assertIs(fields["currency"]["needs_review"], True)

    def test_parser_returns_complete_reviewed_fields_for_blank_input(self) -> None:
        fields = parsing.parse_receipt_fields(())

        self.assertEqual(tuple(fields), EXPECTED_FIELD_NAMES)
        for name in EXPECTED_FIELD_NAMES:
            self._assert_field_contract(fields[name])
            self.assertIsNone(fields[name]["value"])
            self.assertIs(fields[name]["needs_review"], True)

    def test_parser_is_deterministic_for_identical_ocr_evidence(self) -> None:
        lines = (
            self._line(0, "CAFE NINE", top=28),
            self._line(1, "14/03/2026", top=90),
            self._line(2, "TOTAL INR 480.00", top=340),
        )
        self.assertEqual(parsing.parse_receipt_fields(lines), parsing.parse_receipt_fields(lines))

    def test_labelled_rows_beat_row_indexes_and_noisy_duplicate_summary_values(self) -> None:
        lines = (
            self._custom_line(
                0,
                ("2", 97.0),
                ("merchant_name", 92.0),
                ("ShopkKart", 46.0),
                ("Online", 96.0),
                ("Pvt.", 90.0),
                ("Ltd.", 95.0),
            ),
            self._custom_line(
                1,
                ("Merchant", 0.0),
                ("Name", 0.0),
                (":", 73.0),
                ("ShopKart", 73.0),
                ("Online", 96.0),
                ("Pvt.", 94.0),
                ("Ltd.", 94.0),
                ("(E>", 54.0),
            ),
            self._custom_line(
                2,
                ("2", 97.0),
                ("total_amount", 92.0),
                ("%", 79.0),
                ("1,250.00", 93.0),
            ),
            self._custom_line(
                3,
                ("Total", 96.0),
                ("Amount", 96.0),
                (":", 89.0),
                ("21,250.00", 0.0),
                ("sr", 14.0),
            ),
        )

        fields = parsing.parse_receipt_fields(lines)

        self.assertEqual(fields["merchant_name"]["value"], "ShopKart Online Pvt. Ltd.")
        # The near-duplicate spelling remains explicitly reviewed rather than
        # becoming a confident correction, while the selected value is useful.
        self.assertIs(fields["merchant_name"]["needs_review"], True)
        self.assertEqual(fields["total_amount"]["value"], 1250.0)
        self.assertIs(fields["total_amount"]["needs_review"], False)

    def _line(self, index: int, text: str, *, top: int) -> Any:
        tokens = []
        left = 28
        for word in text.split():
            tokens.append(
                parsing.OCRToken(
                    text=word,
                    confidence=96.0,
                    left=left,
                    top=top,
                    width=max(12, len(word) * 13),
                    height=24,
                    block_num=1,
                    par_num=1,
                    line_num=index + 1,
                )
            )
            left += max(12, len(word) * 13) + 10
        return parsing.OCRLine(text=text, tokens=tuple(tokens), index=index)

    def _custom_line(self, index: int, *words: tuple[str, float]) -> Any:
        tokens = []
        left = 28
        for word, confidence in words:
            tokens.append(
                parsing.OCRToken(
                    text=word,
                    confidence=confidence,
                    left=left,
                    top=28 + index * 52,
                    width=max(12, len(word) * 13),
                    height=24,
                    block_num=1,
                    par_num=1,
                    line_num=index + 1,
                )
            )
            left += max(12, len(word) * 13) + 10
        return parsing.OCRLine(
            text=" ".join(word for word, _ in words),
            tokens=tuple(tokens),
            index=index,
        )

    def _assert_field_contract(self, field: Any) -> None:
        self.assertIsInstance(field, Mapping)
        self.assertEqual(set(field), {"value", "confidence", "needs_review"})
        self.assertIsInstance(field["confidence"], float)
        self.assertTrue(math.isfinite(field["confidence"]))
        self.assertGreaterEqual(field["confidence"], 0.0)
        self.assertLessEqual(field["confidence"], 1.0)
        self.assertIsInstance(field["needs_review"], bool)


if __name__ == "__main__":  # pragma: no cover - convenient direct invocation
    unittest.main()
