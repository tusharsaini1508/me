"""Small geometry assertions shared by synthetic receipt tests.

These helpers deliberately live in ``tests`` rather than production code.  They
describe measurable properties of a fixture (area, edge contact, reprojection)
without imposing an implementation strategy on the receipt locator.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def as_quad(points: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Return four finite ``float32`` points, rejecting malformed input."""

    quad = np.asarray(points, dtype=np.float32)
    if quad.shape != (4, 2):
        raise ValueError(f"A quadrilateral must have shape (4, 2), got {quad.shape}.")
    if not np.isfinite(quad).all():
        raise ValueError("A quadrilateral must contain only finite coordinates.")
    return quad


def polygon_area(points: Sequence[Sequence[float]] | np.ndarray) -> float:
    """Return the unsigned shoelace area of a polygon in pixel units."""

    polygon = np.asarray(points, dtype=np.float64)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
        raise ValueError("A polygon must have at least three two-dimensional points.")
    x_coordinates = polygon[:, 0]
    y_coordinates = polygon[:, 1]
    return float(
        0.5
        * abs(
            np.dot(x_coordinates, np.roll(y_coordinates, -1))
            - np.dot(y_coordinates, np.roll(x_coordinates, -1))
        )
    )


def quad_area_ratio(
    quad: Sequence[Sequence[float]] | np.ndarray, image_shape: Sequence[int]
) -> float:
    """Return receipt-quad area divided by the image area."""

    height, width = _height_width(image_shape)
    return polygon_area(as_quad(quad)) / float(height * width)


def touches_image_edge(
    quad: Sequence[Sequence[float]] | np.ndarray,
    image_shape: Sequence[int],
    *,
    tolerance: float = 1.0,
) -> bool:
    """Whether any corner is on or outside the image boundary.

    The helper treats an out-of-frame point as edge contact, which mirrors the
    conservative policy expected from a receipt-capture quality gate.
    """

    if tolerance < 0:
        raise ValueError("tolerance must be non-negative.")
    height, width = _height_width(image_shape)
    corners = as_quad(quad)
    return bool(
        np.any(corners[:, 0] <= tolerance)
        or np.any(corners[:, 1] <= tolerance)
        or np.any(corners[:, 0] >= (width - 1 - tolerance))
        or np.any(corners[:, 1] >= (height - 1 - tolerance))
    )


def mean_corner_distance(
    actual: Sequence[Sequence[float]] | np.ndarray,
    expected: Sequence[Sequence[float]] | np.ndarray,
) -> float:
    """Return mean pointwise distance for equally ordered quadrilaterals."""

    distances = np.linalg.norm(as_quad(actual) - as_quad(expected), axis=1)
    return float(np.mean(distances))


def _height_width(image_shape: Sequence[int]) -> tuple[int, int]:
    if len(image_shape) < 2:
        raise ValueError("image_shape must contain at least height and width.")
    height, width = int(image_shape[0]), int(image_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive.")
    return height, width
