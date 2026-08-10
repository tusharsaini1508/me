"""Contract and geometry tests for the shared headless pipeline.

Tesseract is mocked here on purpose: these tests cover the project's control
flow deterministically on every machine, including contributors who have not
installed the system OCR binary yet.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from src import pipeline
from src.parsing import OCRLine, OCRToken
from tests.fixtures.synthetic_receipts import make_receipt_scene
from tests.geometry import mean_corner_distance


class PipelineContractTests(unittest.TestCase):
    def test_missing_image_returns_complete_safe_failure(self) -> None:
        result = pipeline.extract("this-file-does-not-exist.png")

        self._assert_result_contract(result, passed=False)
        self.assertTrue(result["quality"]["issues"])
        self.assertIn("could not be found", result["quality"]["issues"][0].casefold())

    def test_valid_synthetic_scene_is_localized_and_perspective_rectified(self) -> None:
        scene = make_receipt_scene("valid")

        detection = pipeline.find_receipt_quad(scene.image)

        self.assertIsNotNone(detection)
        assert detection is not None
        self.assertGreater(detection.area_ratio, 0.30)
        self.assertFalse(detection.touches_edge)
        self.assertLess(mean_corner_distance(detection.quad, scene.quad), 18.0)
        rectified = pipeline.rectify_receipt(scene.image, detection)
        self.assertIsNotNone(rectified)
        assert rectified is not None
        self.assertEqual(rectified.ndim, 3)
        self.assertLessEqual(max(rectified.shape[:2]), pipeline.MAX_RECTIFIED_DIMENSION)
        self.assertLessEqual(rectified.shape[0] * rectified.shape[1], pipeline.MAX_RECTIFIED_PIXELS)

    def test_small_and_cutoff_synthetic_scenes_are_rejected_for_geometry(self) -> None:
        small = pipeline.find_receipt_quad(make_receipt_scene("small").image)
        cutoff = pipeline.find_receipt_quad(make_receipt_scene("cutoff").image)

        self.assertIsNotNone(small)
        self.assertIsNotNone(cutoff)
        assert small is not None and cutoff is not None
        self.assertLess(small.area_ratio, pipeline.MIN_RECEIPT_AREA_RATIO)
        self.assertTrue(cutoff.touches_edge)
        self.assertIn("too small", " ".join(pipeline.assess_detection(small)).casefold())
        self.assertIn("cut off", " ".join(pipeline.assess_detection(cutoff)).casefold())

    def test_quality_rejection_never_invokes_ocr(self) -> None:
        dark = np.zeros((720, 960, 3), dtype=np.uint8)
        with self._temporary_png(dark) as path, patch.object(pipeline, "ocr_lines") as mocked_ocr:
            result = pipeline.extract(str(path))

        self._assert_result_contract(result, passed=False)
        mocked_ocr.assert_not_called()
        self.assertIn("underexposed", " ".join(result["quality"]["issues"]).casefold())

    def test_low_resolution_image_is_rejected_before_expensive_processing(self) -> None:
        low_resolution = np.full((300, 420, 3), 180, dtype=np.uint8)
        with self._temporary_png(low_resolution) as path, patch.object(pipeline, "find_receipt_quad") as mocked_find:
            result = pipeline.extract(str(path))

        self._assert_result_contract(result, passed=False)
        mocked_find.assert_not_called()
        self.assertIn("resolution is too low", " ".join(result["quality"]["issues"]).casefold())

    def test_exposure_and_glare_signals_are_actionable(self) -> None:
        clipped_white = np.full((720, 960, 3), 255, dtype=np.uint8)

        metrics = pipeline.measure_quality(clipped_white)
        issues = " ".join(pipeline.assess_quality(metrics)).casefold()

        self.assertGreater(metrics.clipped_bright_fraction, 0.99)
        self.assertGreater(metrics.glare_fraction, 0.99)
        self.assertIn("overexposed", issues)
        self.assertIn("glare", issues)

    def test_sharp_full_frame_document_is_not_mistaken_for_glare(self) -> None:
        document = np.full((720, 960, 3), 255, dtype=np.uint8)
        cv2.rectangle(document, (8, 8), (951, 711), (35, 61, 107), 8)
        for row in range(120, 610, 55):
            cv2.putText(
                document,
                "PAYMENT RECEIPT  INR 1,250.00",
                (55, row),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (35, 61, 107),
                2,
                cv2.LINE_AA,
            )

        metrics = pipeline.measure_quality(document)
        detection = pipeline.find_receipt_quad(document)

        self.assertEqual(metrics.glare_fraction, 0.0)
        self.assertNotIn("glare", " ".join(pipeline.assess_quality(metrics)).casefold())
        self.assertIsNotNone(detection)
        assert detection is not None
        self.assertEqual(detection.area_ratio, 1.0)
        self.assertFalse(detection.touches_edge)

    def test_tesseract_resolver_honors_an_explicit_existing_path(self) -> None:
        class Backend:
            tesseract_cmd = "missing-tesseract"

        class Module:
            pytesseract = Backend

        with tempfile.TemporaryDirectory() as temporary_directory:
            executable = Path(temporary_directory) / "tesseract.exe"
            executable.touch()
            with patch.dict(pipeline.os.environ, {"TESSERACT_CMD": str(executable)}, clear=False):
                resolved = pipeline.resolve_tesseract_command(Module)

        self.assertEqual(resolved, str(executable.resolve()))
        self.assertEqual(Backend.tesseract_cmd, str(executable.resolve()))

    def test_unavailable_ocr_fails_closed_after_a_good_capture(self) -> None:
        scene = make_receipt_scene("valid")
        with self._temporary_png(scene.image) as path, patch.object(
            pipeline, "ocr_lines", side_effect=pipeline.OCRUnavailableError("missing")
        ):
            result = pipeline.extract(str(path))

        self._assert_result_contract(result, passed=False)
        self.assertIn("local ocr is unavailable", " ".join(result["quality"]["issues"]).casefold())

    def test_good_capture_uses_typed_ocr_lines_and_public_parser(self) -> None:
        scene = make_receipt_scene("valid")
        lines = self._clear_ocr_lines()
        with self._temporary_png(scene.image) as path, patch.object(pipeline, "ocr_lines", return_value=lines):
            result = pipeline.extract(str(path))

        self._assert_result_contract(result, passed=True)
        self.assertEqual(result["fields"]["merchant_name"]["value"], "CAFE NINE")
        self.assertEqual(result["fields"]["transaction_date"]["value"], "2026-03-14")
        self.assertEqual(result["fields"]["total_amount"]["value"], 480.0)
        self.assertEqual(result["fields"]["currency"]["value"], "INR")
        self.assertIs(result["fields"]["total_amount"]["needs_review"], False)

    def test_preview_returns_none_for_invalid_geometry_and_bgr_for_valid_scene(self) -> None:
        valid = make_receipt_scene("valid").image
        cutoff = make_receipt_scene("cutoff").image
        with self._temporary_png(valid) as valid_path, self._temporary_png(cutoff) as cutoff_path:
            preview = pipeline.rectify_preview(str(valid_path))
            rejected_preview = pipeline.rectify_preview(str(cutoff_path))

        self.assertIsNotNone(preview)
        assert preview is not None
        self.assertEqual(preview.shape[2], 3)
        self.assertIsNone(rejected_preview)

    def test_large_working_image_is_downscaled_before_processing(self) -> None:
        source = np.zeros((2_000, 2_500, 3), dtype=np.uint8)

        resized, scale = pipeline._resize_bounded(source)

        self.assertLess(scale, 1.0)
        self.assertLessEqual(max(resized.shape[:2]), pipeline.MAX_WORKING_DIMENSION)
        self.assertLessEqual(resized.shape[0] * resized.shape[1], pipeline.MAX_WORKING_PIXELS)

    def _clear_ocr_lines(self) -> list[OCRLine]:
        values = (
            "CAFE NINE",
            "14/03/2026 11:42",
            "Masala chai INR 80.00",
            "TOTAL INR 480.00",
        )
        lines: list[OCRLine] = []
        for index, text in enumerate(values):
            left = 20
            tokens: list[OCRToken] = []
            for word in text.split():
                tokens.append(
                    OCRToken(
                        text=word,
                        confidence=96.0,
                        left=left,
                        top=30 + index * 62,
                        width=max(12, len(word) * 12),
                        height=22,
                        block_num=1,
                        par_num=1,
                        line_num=index + 1,
                    )
                )
                left += max(12, len(word) * 12) + 9
            lines.append(OCRLine(text=text, tokens=tuple(tokens), index=index, top=30 + index * 62))
        return lines

    def _temporary_png(self, image: np.ndarray):
        temporary_directory = tempfile.TemporaryDirectory()
        path = Path(temporary_directory.name) / "receipt.png"
        if not cv2.imwrite(str(path), image):
            temporary_directory.cleanup()
            raise OSError("Could not create temporary image fixture.")

        class _PathContext:
            def __enter__(self) -> Path:
                return path

            def __exit__(self, *_: object) -> None:
                temporary_directory.cleanup()

        return _PathContext()

    def _assert_result_contract(self, result: object, *, passed: bool) -> None:
        self.assertIsInstance(result, dict)
        assert isinstance(result, dict)
        self.assertEqual(set(result), {"quality", "fields", "processing_ms"})
        self.assertIs(result["quality"]["pass"], passed)
        self.assertIsInstance(result["quality"]["issues"], list)
        self.assertEqual(tuple(result["fields"]), ("merchant_name", "transaction_date", "total_amount", "currency"))
        self.assertGreaterEqual(result["processing_ms"], 0)
        for field in result["fields"].values():
            self.assertEqual(set(field), {"value", "confidence", "needs_review"})
            self.assertIsInstance(field["confidence"], float)
            self.assertGreaterEqual(field["confidence"], 0.0)
            self.assertLessEqual(field["confidence"], 1.0)
            self.assertIsInstance(field["needs_review"], bool)
        self.assertIsInstance(json.dumps(result, allow_nan=False), str)


if __name__ == "__main__":  # pragma: no cover - convenient direct invocation
    unittest.main()
