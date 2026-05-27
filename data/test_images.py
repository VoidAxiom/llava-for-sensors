from __future__ import annotations

from itertools import combinations
from pathlib import Path
import io

import numpy as np
from PIL import Image
import pytest

from data.images import (
    CLASS_NAMES,
    IMAGE_SIZE,
    get_image_for_label,
    render_class_image,
    write_class_images_to_disk,
)


def test_render_returns_correct_size() -> None:
    image = render_class_image("normal")

    assert image.size == IMAGE_SIZE
    assert image.mode == "RGB"


def test_determinism() -> None:
    first = render_class_image("normal")
    second = render_class_image("normal")

    assert _png_bytes(first) == _png_bytes(second)


def test_all_classes_render() -> None:
    for class_name in CLASS_NAMES:
        image = render_class_image(class_name)

        assert image.size == IMAGE_SIZE
        assert image.mode == "RGB"


def test_get_image_for_label() -> None:
    image = get_image_for_label(0)
    expected = render_class_image("normal")

    assert _png_bytes(image) == _png_bytes(expected)


def test_label_out_of_range() -> None:
    with pytest.raises((IndexError, ValueError)):
        get_image_for_label(99)


def test_write_to_disk(tmp_path: Path) -> None:
    write_class_images_to_disk(tmp_path)

    for class_name in CLASS_NAMES:
        path = tmp_path / f"{class_name}.png"
        assert path.exists()
        with Image.open(path) as image:
            assert image.size == IMAGE_SIZE
            assert image.mode == "RGB"


def test_class_images_distinct() -> None:
    images = {class_name: render_class_image(class_name) for class_name in CLASS_NAMES}

    for left_name, right_name in combinations(CLASS_NAMES, 2):
        left = images[left_name]
        right = images[right_name]
        assert _pixel_mse(left, right) > 0.0


def test_determinism_all_classes() -> None:
    for class_name in CLASS_NAMES:
        first = render_class_image(class_name)
        second = render_class_image(class_name)

        assert _png_bytes(first) == _png_bytes(second)


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()


def _pixel_mse(left: Image.Image, right: Image.Image) -> float:
    left_pixels = np.asarray(left, dtype=np.float32)
    right_pixels = np.asarray(right, dtype=np.float32)
    delta = left_pixels - right_pixels
    return float(np.mean(delta * delta))
