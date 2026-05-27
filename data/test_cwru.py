"""Tests for data/cwru.py — uses synthetic .mat fixtures under data/test_assets/cwru/."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.io

from data.cwru import (
    CLASS_LABELS,
    CLASS_NAMES,
    CLASS_NATIVE_RATE_HZ,
    SAMPLE_RATE_HZ,
    WINDOW_SIZE,
    build_split,
    load_class_windows,
    load_cwru_mat,
    preprocess_to_windows,
)

FIXTURE_ROOT = Path(__file__).parent / "test_assets" / "cwru"


@pytest.fixture
def fixture_root() -> Path:
    return FIXTURE_ROOT


def test_load_cwru_mat_probes_key(tmp_path: Path) -> None:
    """load_cwru_mat finds X*_DE_time key and returns float32 array."""
    signal = np.arange(4096, dtype=np.float64)
    mat_path = tmp_path / "test.mat"
    scipy.io.savemat(str(mat_path), {"X097_DE_time": signal})
    result = load_cwru_mat(mat_path)
    assert result.shape == (4096,)
    assert result.dtype == np.float32


def test_load_cwru_mat_resamples_48khz_to_12khz(tmp_path: Path) -> None:
    """load_cwru_mat with native_rate_hz=48000 downsamples 4:1."""
    signal = np.arange(48000, dtype=np.float64)
    mat_path = tmp_path / "normal.mat"
    scipy.io.savemat(str(mat_path), {"X097_DE_time": signal})
    result = load_cwru_mat(mat_path, native_rate_hz=48000)
    assert result.shape == (12000,)
    assert result.dtype == np.float32


def test_load_cwru_mat_raises_on_missing_key(tmp_path: Path) -> None:
    """load_cwru_mat raises ValueError when no X*_DE_time key is present."""
    mat_path = tmp_path / "bad.mat"
    scipy.io.savemat(str(mat_path), {"irrelevant_key": np.zeros(100)})
    with pytest.raises(ValueError, match="X\\*_DE_time"):
        load_cwru_mat(mat_path)


def test_preprocess_to_windows_basic() -> None:
    """5000-sample signal → 2 windows of 2048 (remainder dropped)."""
    signal = np.ones(5000, dtype=np.float32)
    windows = preprocess_to_windows(signal, window_size=2048)
    assert windows.shape == (2, 2048)
    assert windows.dtype == np.float32


def test_preprocess_to_windows_exact_multiple() -> None:
    """4096-sample signal → exactly 2 windows of 2048."""
    signal = np.ones(4096, dtype=np.float32)
    windows = preprocess_to_windows(signal, window_size=2048)
    assert windows.shape == (2, 2048)
    assert windows.dtype == np.float32


def test_load_class_windows(fixture_root: Path) -> None:
    """Fixture: 4 files × 16384 samples each → 4 × 8 = 32 windows per class."""
    windows = load_class_windows(fixture_root / "normal")
    assert windows.shape == (32, 2048)
    assert windows.dtype == np.float32


def test_load_class_windows_passes_native_rate(fixture_root: Path) -> None:
    """native_rate_hz=48000 → 4:1 downsample → fewer windows per file."""
    # Each fixture .mat has 16384 samples (at "12kHz equivalent").
    # With native_rate_hz=48000, signal is resampled to 16384//4=4096 samples.
    # 4096 samples // 2048 window_size = 2 windows per file × 4 files = 8 windows.
    windows = load_class_windows(fixture_root / "normal", native_rate_hz=48000)
    assert windows.shape == (8, 2048)
    assert windows.dtype == np.float32


def test_build_split_deterministic(fixture_root: Path) -> None:
    """Two calls with seed=0 produce byte-identical arrays."""
    s1 = build_split(fixture_root, seed=0)
    s2 = build_split(fixture_root, seed=0)
    assert np.array_equal(s1["train"][0], s2["train"][0]), "X_train not identical"
    assert np.array_equal(s1["train"][1], s2["train"][1]), "y_train not identical"
    assert s1["train"][1].dtype == np.int64


def test_build_split_proportions(fixture_root: Path) -> None:
    """104 total windows, file-grouped split preserves rough proportions."""
    split = build_split(fixture_root, seed=0)
    y_train = split["train"][1]
    y_val = split["val"][1]
    y_test = split["test"][1]
    total = len(y_train) + len(y_val) + len(y_test)
    assert total == 104
    assert 70 <= len(y_train) <= 90, f"y_train size {len(y_train)} not in [70, 90]"
    assert len(y_val) >= 8
    assert len(y_test) >= 8


def test_class_labels_and_native_rates() -> None:
    """CLASS_NATIVE_RATE_HZ covers all CLASS_NAMES with correct rates."""
    assert set(CLASS_NATIVE_RATE_HZ.keys()) == set(CLASS_NAMES)
    assert CLASS_NATIVE_RATE_HZ["normal"] == 48000
    for cls in ("inner_race", "outer_race", "ball"):
        assert CLASS_NATIVE_RATE_HZ[cls] == SAMPLE_RATE_HZ


def test_class_labels_canonical() -> None:
    """Constants match the pinned spec."""
    assert CLASS_LABELS == {"normal": 0, "inner_race": 1, "outer_race": 2, "ball": 3}
    assert CLASS_NAMES == ("normal", "inner_race", "outer_race", "ball")
    assert WINDOW_SIZE == 2048
    assert SAMPLE_RATE_HZ == 12000
