from __future__ import annotations

import pytest

from data.notes import CLASS_NAMES, synthesize_note, synthesize_note_for_class


def test_normal_note_content() -> None:
    note = synthesize_note(0)

    assert "Routine inspection" in note
    assert "baseline" in note


def test_determinism() -> None:
    assert synthesize_note(0) == synthesize_note(0)


def test_all_classes_produce_notes() -> None:
    for label in range(len(CLASS_NAMES)):
        assert synthesize_note(label)


def test_classes_produce_distinct_notes() -> None:
    notes = {synthesize_note(label) for label in range(len(CLASS_NAMES))}

    assert len(notes) == len(CLASS_NAMES)


def test_load_hp_substitution() -> None:
    assert "3 HP load" in synthesize_note(1, load_hp=3)


def test_fault_diameter_substitution() -> None:
    assert "0.021 in" in synthesize_note(2, fault_diameter_in=0.021)


def test_synthesize_note_for_class_matches_label() -> None:
    assert synthesize_note_for_class("normal") == synthesize_note(0)


def test_invalid_label() -> None:
    with pytest.raises((IndexError, ValueError)):
        synthesize_note(99)


def test_invalid_class_name() -> None:
    with pytest.raises((KeyError, ValueError)):
        synthesize_note_for_class("unknown")
