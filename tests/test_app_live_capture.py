import unittest

import app


class LiveCaptureTests(unittest.TestCase):
    def test_quality_passed_result_is_accepted(self) -> None:
        result = {"quality": {"pass": True, "issues": []}}
        self.assertTrue(app._should_accept_capture(result))

    def test_quality_failed_result_is_rejected(self) -> None:
        result = {"quality": {"pass": False, "issues": ["Image is too blurry"]}}
        self.assertFalse(app._should_accept_capture(result))

    def test_missing_quality_is_rejected(self) -> None:
        result = {"fields": {}}
        self.assertFalse(app._should_accept_capture(result))


if __name__ == "__main__":
    unittest.main()
