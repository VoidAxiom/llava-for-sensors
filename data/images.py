"""Procedural bearing-diagram images for fault classes."""

from __future__ import annotations

from math import cos, pi, sin
from pathlib import Path

from PIL import Image, ImageDraw

CLASS_NAMES: tuple[str, ...] = ("normal", "inner_race", "outer_race", "ball")
IMAGE_SIZE: tuple[int, int] = (224, 224)
ASSETS_DIR: Path = Path(__file__).parent / "assets" / "images"

BG = (255, 255, 255)
OUTER_RING = (64, 64, 64)
INNER_RACE = (128, 128, 128)
BALL = (160, 160, 160)
FAULT_RED = (220, 50, 50)
CENTER = (200, 200, 200)

_CENTER_X = 112
_CENTER_Y = 112
_BALL_COUNT = 8
_BALL_TRACK_RADIUS = 57
_BALL_RADIUS = 10
_FAULT_RADIUS = 8


def render_class_image(class_name: str) -> Image.Image:
    """Return the canonical PIL.Image for a class. Deterministic — same name → same image."""

    if class_name not in CLASS_NAMES:
        raise ValueError(f"class_name must be one of {CLASS_NAMES}; got {class_name!r}")

    image = Image.new("RGB", IMAGE_SIZE, BG)
    draw = ImageDraw.Draw(image)

    _draw_annulus(draw, outer_radius=90, inner_radius=70, fill=OUTER_RING, inner_fill=BG)
    _draw_balls(draw, faulted_ball=class_name == "ball")
    _draw_annulus(
        draw,
        outer_radius=45,
        inner_radius=25,
        fill=INNER_RACE,
        inner_fill=CENTER,
    )

    if class_name == "outer_race":
        _draw_fault_marker(draw, center=(_CENTER_X, _CENTER_Y - 90))
    elif class_name == "inner_race":
        _draw_fault_marker(draw, center=(_CENTER_X, _CENTER_Y - 45))

    return image


def get_image_for_label(label: int) -> Image.Image:
    """Convenience: label int → PIL.Image. Calls render_class_image(CLASS_NAMES[label])."""

    return render_class_image(_class_name_for_label(label))


def write_class_images_to_disk(out_dir: Path = ASSETS_DIR) -> None:
    """Render all 4 class images and save as PNG under out_dir/{class_name}.png. Idempotent."""

    out_dir.mkdir(parents=True, exist_ok=True)
    for class_name in CLASS_NAMES:
        path = out_dir / f"{class_name}.png"
        render_class_image(class_name).save(path, format="PNG", optimize=False)


def _class_name_for_label(label: int) -> str:
    if label < 0 or label >= len(CLASS_NAMES):
        raise ValueError(f"label must be in 0..{len(CLASS_NAMES) - 1}; got {label}")
    return CLASS_NAMES[label]


def _draw_annulus(
    draw: ImageDraw.ImageDraw,
    *,
    outer_radius: int,
    inner_radius: int,
    fill: tuple[int, int, int],
    inner_fill: tuple[int, int, int],
) -> None:
    draw.ellipse(_bbox(outer_radius), fill=fill)
    draw.ellipse(_bbox(inner_radius), fill=inner_fill)


def _draw_balls(draw: ImageDraw.ImageDraw, *, faulted_ball: bool) -> None:
    for index in range(_BALL_COUNT):
        angle = -pi / 2 + index * (2 * pi / _BALL_COUNT)
        x = round(_CENTER_X + _BALL_TRACK_RADIUS * cos(angle))
        y = round(_CENTER_Y + _BALL_TRACK_RADIUS * sin(angle))
        fill = FAULT_RED if faulted_ball and index == 0 else BALL
        draw.ellipse(_bbox(_BALL_RADIUS, center=(x, y)), fill=fill)


def _draw_fault_marker(draw: ImageDraw.ImageDraw, *, center: tuple[int, int]) -> None:
    draw.ellipse(_bbox(_FAULT_RADIUS, center=center), fill=FAULT_RED)


def _bbox(
    radius: int,
    *,
    center: tuple[int, int] = (_CENTER_X, _CENTER_Y),
) -> tuple[int, int, int, int]:
    x, y = center
    return (x - radius, y - radius, x + radius, y + radius)
