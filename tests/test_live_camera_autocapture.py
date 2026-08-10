"""Tests for the live-camera auto-capture integration helpers."""

from __future__ import annotations

import unittest

import app


class LiveCameraAutoCaptureTests(unittest.TestCase):
    def test_component_payload_is_decoded_to_bytes_and_filename(self) -> None:
        payload = "data:image/jpeg;base64,AAECAw=="
        image_bytes, name = app._decode_component_payload(payload)

        self.assertEqual(name, "live-capture.jpg")
        self.assertEqual(image_bytes, b"\x00\x01\x02\x03")

    def test_empty_component_payload_is_ignored(self) -> None:
        self.assertEqual(app._decode_component_payload(None), (None, None))
        self.assertEqual(app._decode_component_payload(""), (None, None))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
