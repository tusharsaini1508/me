"""Deterministic, rights-clear receipt scenes for vision and contract tests.

The generator intentionally produces simple geometric receipts instead of
pretending to be a realistic OCR benchmark.  It gives tests repeatable coverage
for a valid perspective crop, a too-small receipt, and a receipt cut off by the
frame.  Real receipts belong in the separately curated ``test/`` evaluation
set, not in this unit-test fixture module.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import cv2
import numpy as np


SceneKind = Literal["valid", "small", "cutoff"]

DEFAULT_FIELDS: Final[dict[str, object]] = {
    "merchant_name": "Cafe Nine",
    "transaction_date": "2026-03-14",
    "total_amount": 480.0,
    "currency": "INR",
}


@dataclass(frozen=True)
class SyntheticReceiptScene:
    """An in-memory fixture and the known receipt corners used to make it."""

    image: np.ndarray
    quad: np.ndarray
    fields: dict[str, object]
    kind: SceneKind


_NORMALISED_QUADS: Final[dict[SceneKind, np.ndarray]] = {
    # Top-left, top-right, bottom-right, bottom-left.  The valid card fills a
    # useful portion of the frame but leaves visible table around every edge.
    "valid": np.array(
        ((0.19, 0.11), (0.81, 0.16), (0.74, 0.89), (0.22, 0.84)), dtype=np.float32
    ),
    # A readable-looking card that intentionally occupies too little frame area.
    "small": np.array(
        ((0.42, 0.35), (0.59, 0.37), (0.57, 0.65), (0.44, 0.63)), dtype=np.float32
    ),
    # Two corners lie outside the left image edge, exercising cut-off handling.
    "cutoff": np.array(
        ((-0.06, 0.13), (0.78, 0.16), (0.73, 0.89), (-0.04, 0.85)), dtype=np.float32
    ),
}


def make_receipt_scene(
    kind: SceneKind = "valid", *, width: int = 960, height: int = 720
) -> SyntheticReceiptScene:
    """Build a BGR ``uint8`` receipt scene without randomness or external files."""

    if kind not in _NORMALISED_QUADS:
        raise ValueError(f"Unsupported synthetic scene kind: {kind!r}.")
    if width < 160 or height < 160:
        raise ValueError("Synthetic scenes must be at least 160 by 160 pixels.")

    card = _receipt_card()
    canvas = _table_background(width, height)
    quad = _scale_quad(_NORMALISED_QUADS[kind], width=width, height=height)
    source = np.array(
        ((0, 0), (card.shape[1] - 1, 0), (card.shape[1] - 1, card.shape[0] - 1), (0, card.shape[0] - 1)),
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(source, quad)
    warped_card = cv2.warpPerspective(
        card,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    card_mask = cv2.warpPerspective(
        np.full(card.shape[:2], 255, dtype=np.uint8),
        transform,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    canvas[card_mask > 0] = warped_card[card_mask > 0]

    return SyntheticReceiptScene(
        image=canvas,
        quad=quad.copy(),
        fields=dict(DEFAULT_FIELDS),
        kind=kind,
    )


def write_fixture_set(directory: str | Path) -> dict[str, Path]:
    """Write a tiny generated fixture set and labels for local smoke testing.

    Only the normal capture is labelled, because the small and cut-off images
    are capture-quality negatives rather than extraction-accuracy examples.
    """

    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for kind in ("valid", "small", "cutoff"):
        path = destination / f"synthetic_{kind}_receipt.png"
        if not cv2.imwrite(str(path), make_receipt_scene(kind).image):
            raise OSError(f"Could not write synthetic fixture: {path}")
        written[kind] = path

    labels_path = destination / "labels.json"
    labels = {
        "receipts": [
            {
                "image": written["valid"].name,
                "fields": dict(DEFAULT_FIELDS),
            }
        ]
    }
    labels_path.write_text(json.dumps(labels, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    written["labels"] = labels_path
    return written


def _table_background(width: int, height: int) -> np.ndarray:
    """Create a lightly textured dark tabletop with a deterministic gradient."""

    y_coordinates, x_coordinates = np.indices((height, width))
    texture = ((x_coordinates * 3 + y_coordinates * 5) % 13).astype(np.uint8)
    background = np.empty((height, width, 3), dtype=np.uint8)
    background[..., 0] = 51 + texture // 3
    background[..., 1] = 67 + texture // 2
    background[..., 2] = 74 + texture
    return background


def _receipt_card() -> np.ndarray:
    """Create a high-contrast fake receipt before perspective projection."""

    width, height = 520, 780
    card = np.full((height, width, 3), (244, 246, 242), dtype=np.uint8)
    cv2.rectangle(card, (5, 5), (width - 6, height - 6), (138, 138, 138), 2)
    _text(card, "CAFE NINE", (42, 82), 1.25, 3)
    _text(card, "14/03/2026   11:42", (42, 132), 0.60, 1)
    _text(card, "Masala chai", (44, 245), 0.66, 1)
    _text(card, "INR  80.00", (310, 245), 0.66, 1)
    _text(card, "Lunch bowl", (44, 298), 0.66, 1)
    _text(card, "INR 400.00", (298, 298), 0.66, 1)
    cv2.line(card, (40, 350), (480, 350), (55, 55, 55), 2)
    _text(card, "TOTAL", (44, 420), 0.90, 2)
    _text(card, "INR 480.00", (256, 420), 0.90, 2)
    _text(card, "Thank you", (156, 690), 0.64, 1)
    return card


def _text(image: np.ndarray, text: str, origin: tuple[int, int], scale: float, thickness: int) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (35, 35, 35),
        thickness,
        lineType=cv2.LINE_AA,
    )


def _scale_quad(normalised_quad: np.ndarray, *, width: int, height: int) -> np.ndarray:
    scale = np.array((width - 1, height - 1), dtype=np.float32)
    return (normalised_quad * scale).astype(np.float32)
