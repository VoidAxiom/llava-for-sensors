"""Synthetic cross-modal toy data for sensor, image, and text fusion."""

from __future__ import annotations

from typing import TypedDict

import numpy as np

N_CLASSES = 4
SENSOR_LENGTH = 2048
SAMPLE_RATE_HZ = 12000.0
IMAGE_SHAPE = (224, 224, 3)
LOW_FREQUENCIES_HZ = (120.0, 180.0)
HIGH_FREQUENCIES_HZ = (240.0, 360.0)


class SyntheticSample(TypedDict):
    """One generated toy-dataset sample."""

    sensor: np.ndarray
    image: np.ndarray
    text: str
    label: int


TEXT_TEMPLATES: dict[int, tuple[str, ...]] = {
    0: (
        "Vibration envelope smooth. Temperature normal.",
        "Sensor note: rounded vibration trace with nominal thermal reading.",
        "Technician note: steady vibration envelope. Temperature within limits.",
    ),
    1: (
        "Vibration envelope abrupt. Temperature elevated.",
        "Sensor note: squared vibration trace with thermal reading above limits.",
        "Technician note: sharp vibration envelope. Temperature warning.",
    ),
    2: (
        "Vibration envelope smooth. Temperature normal.",
        "Sensor note: rounded vibration trace with nominal thermal reading.",
        "Technician note: steady vibration envelope. Temperature within limits.",
    ),
    3: (
        "Vibration envelope abrupt. Temperature elevated.",
        "Sensor note: squared vibration trace with thermal reading above limits.",
        "Technician note: sharp vibration envelope. Temperature warning.",
    ),
}


def _validate_class_id(class_id: int) -> None:
    if class_id not in TEXT_TEMPLATES:
        raise ValueError(f"class_id must be in 0..{N_CLASSES - 1}; got {class_id}")


def _sample_sensor(class_id: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a clean noisy sine wave with class-pair frequency ambiguity."""

    _validate_class_id(class_id)
    t = np.arange(SENSOR_LENGTH, dtype=np.float32) / SAMPLE_RATE_HZ
    if class_id in (0, 1):
        freq = float(rng.choice(LOW_FREQUENCIES_HZ))
    else:
        freq = float(rng.choice(HIGH_FREQUENCIES_HZ))
    signal = np.sin(2.0 * np.pi * freq * t).astype(np.float32)
    noise = rng.normal(0.0, 0.05, size=SENSOR_LENGTH).astype(np.float32)
    return signal + noise


def _sample_image(class_id: int) -> np.ndarray:
    """Generate an RGB schematic image with class-pair template ambiguity."""

    _validate_class_id(class_id)
    img = np.ones(IMAGE_SHAPE, dtype=np.uint8) * 255
    if class_id in (0, 2):
        yy, xx = np.ogrid[: IMAGE_SHAPE[0], : IMAGE_SHAPE[1]]
        mask = (xx - 112) ** 2 + (yy - 112) ** 2 <= 60**2
        img[mask] = np.array([0, 0, 200], dtype=np.uint8)
    else:
        img[67:157, 67:157] = np.array([200, 0, 0], dtype=np.uint8)
    return img


def _sample_text(class_id: int, rng: np.random.Generator) -> str:
    """Return a technician note with deliberate text-only ambiguity."""

    _validate_class_id(class_id)
    return str(rng.choice(TEXT_TEMPLATES[class_id]))


def _balanced_labels(n: int) -> list[int]:
    if n < 0:
        raise ValueError(f"n must be non-negative; got {n}")
    base_count = n // N_CLASSES
    remainder = n % N_CLASSES
    labels: list[int] = []
    for class_id in range(N_CLASSES):
        labels.extend([class_id] * (base_count + int(class_id < remainder)))
    return labels


def generate(n: int = 1000, seed: int = 0) -> list[SyntheticSample]:
    """Generate deterministic synthetic multimodal samples.

    The labels are balanced as evenly as possible, then shuffled by ``seed``.
    Each single modality is intentionally ambiguous; the joint sensor, image,
    and text signature identifies the class.
    """

    rng = np.random.default_rng(seed)
    labels = _balanced_labels(n)
    rng.shuffle(labels)

    samples: list[SyntheticSample] = []
    for class_id in labels:
        samples.append(
            {
                "sensor": _sample_sensor(class_id, rng),
                "image": _sample_image(class_id),
                "text": _sample_text(class_id, rng),
                "label": class_id,
            }
        )
    return samples
