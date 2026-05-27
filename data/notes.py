"""Template-based technician notes for bearing fault classes."""

from __future__ import annotations

from collections.abc import Mapping

CLASS_NAMES: tuple[str, ...] = ("normal", "inner_race", "outer_race", "ball")

NOTE_TEMPLATES: Mapping[str, str] = {
    "normal": (
        "Routine inspection. {load_hp} HP load. Vibration spectrum within nominal envelope; "
        "no characteristic fault frequencies detected. Bearing condition: baseline."
    ),
    "inner_race": (
        "{load_hp} HP load. High-frequency vibration component detected at the inner-race "
        "characteristic frequency BPFI; sidebands at shaft speed visible. Signature consistent "
        "with inner-race spalling. Severity proxy: fault diameter {fault_diameter_in} in."
    ),
    "outer_race": (
        "{load_hp} HP load. Strong impulse train at the outer-race characteristic frequency "
        "BPFO; no shaft-speed sidebands (stationary defect). Outer-race spalling consistent "
        "with the impulse pattern. Severity proxy: fault diameter {fault_diameter_in} in."
    ),
    "ball": (
        "{load_hp} HP load. Modulated vibration at the ball-spin frequency BSF, double-sided "
        "sidebands at the cage frequency FTF — characteristic of a rolling-element (ball) "
        "defect. Severity proxy: fault diameter {fault_diameter_in} in."
    ),
}


def synthesize_note(label: int, *, load_hp: int = 1, fault_diameter_in: float = 0.007) -> str:
    """Build a technician note from label + metadata. Deterministic — same args → same string verbatim."""

    return synthesize_note_for_class(
        _class_name_for_label(label),
        load_hp=load_hp,
        fault_diameter_in=fault_diameter_in,
    )


def synthesize_note_for_class(
    class_name: str,
    *,
    load_hp: int = 1,
    fault_diameter_in: float = 0.007,
) -> str:
    """Convenience: class_name → note string."""

    template = NOTE_TEMPLATES.get(class_name)
    if template is None:
        raise ValueError(f"class_name must be one of {CLASS_NAMES}; got {class_name!r}")
    return template.format(load_hp=load_hp, fault_diameter_in=fault_diameter_in)


def _class_name_for_label(label: int) -> str:
    if label < 0 or label >= len(CLASS_NAMES):
        raise ValueError(f"label must be in 0..{len(CLASS_NAMES) - 1}; got {label}")
    return CLASS_NAMES[label]
