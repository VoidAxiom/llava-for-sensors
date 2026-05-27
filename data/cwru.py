"""CWRU Bearing Data Center loader, preprocessor, and stratified splitter.

Manual fetch: see README.md § "CWRU dataset — manual fetch".
Expected layout under data/raw/cwru/:
    normal/          (4 records: 97.mat 98.mat 99.mat 100.mat)
    inner_race/      (multiple records)
    outer_race/      (multiple records)
    ball/            (multiple records)
Normal baseline files (97.mat–100.mat) are 48 kHz recordings; pass
native_rate_hz=48000 to load_class_windows for the normal/ class.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path

import numpy as np
import scipy.io
import scipy.signal
from sklearn.model_selection import train_test_split
import torch

CLASS_NAMES: tuple[str, ...] = ("normal", "inner_race", "outer_race", "ball")
CLASS_LABELS: dict[str, int] = {"normal": 0, "inner_race": 1, "outer_race": 2, "ball": 3}
WINDOW_SIZE: int = 2048
SAMPLE_RATE_HZ: int = 12000


def load_cwru_mat(path: Path, native_rate_hz: int = SAMPLE_RATE_HZ) -> np.ndarray:
    """Load a CWRU .mat file's drive-end accelerometer channel as float32."""

    mat_data = scipy.io.loadmat(str(path))
    available_keys = [key for key in mat_data if not key.startswith("_")]
    drive_end_key = next(
        (key for key in available_keys if fnmatch.fnmatch(key, "X*_DE_time")), None
    )
    if drive_end_key is None:
        raise ValueError(
            f"No drive-end key (X*_DE_time) found in {path}. "
            f"Available keys: {sorted(available_keys)}"
        )
    raw = np.asarray(mat_data[drive_end_key]).reshape(-1)
    if native_rate_hz != SAMPLE_RATE_HZ:
        resampled = scipy.signal.resample_poly(raw, SAMPLE_RATE_HZ, native_rate_hz)
        return resampled.astype(np.float32)
    return raw.astype(np.float32)


def preprocess_to_windows(samples: np.ndarray, window_size: int = WINDOW_SIZE) -> np.ndarray:
    """Convert a 1-D signal into non-overlapping fixed-size windows."""

    n_windows = len(samples) // window_size
    return samples[: n_windows * window_size].reshape(n_windows, window_size).astype(np.float32)


def load_class_windows(class_dir: Path, native_rate_hz: int = SAMPLE_RATE_HZ) -> np.ndarray:
    """Load and window every .mat file in one class directory.

    Pass native_rate_hz=48000 for CWRU normal baseline files (97.mat–100.mat).
    """

    if not class_dir.exists():
        raise FileNotFoundError(f"Class directory not found: {class_dir}")
    mat_files = sorted(class_dir.glob("*.mat"))
    if not mat_files:
        raise ValueError(f"No .mat files found in {class_dir}")
    windows = [
        preprocess_to_windows(load_cwru_mat(mat_file, native_rate_hz=native_rate_hz))
        for mat_file in mat_files
    ]
    return np.concatenate(windows, axis=0)


def build_split(raw_root: Path, seed: int = 0) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Build deterministic file-grouped stratified train/val/test splits.

    Splits at the recording (.mat file) level first to prevent data leakage
    from near-duplicate contiguous windows sharing a recording across splits.
    """

    file_windows: list[np.ndarray] = []
    file_labels: list[int] = []
    for class_name in CLASS_NAMES:
        class_dir = raw_root / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Class directory not found: {class_dir}")
        mat_files = sorted(class_dir.glob("*.mat"))
        if not mat_files:
            raise ValueError(f"No .mat files found in {class_dir}")
        label = CLASS_LABELS[class_name]
        for mat_file in mat_files:
            windows = preprocess_to_windows(load_cwru_mat(mat_file))
            file_windows.append(windows)
            file_labels.append(label)

    indices = list(range(len(file_windows)))
    idx_train, idx_hold = train_test_split(
        indices,
        test_size=0.20,
        stratify=file_labels,
        random_state=seed,
    )
    hold_labels = [file_labels[i] for i in idx_hold]
    hold_counts = np.unique(hold_labels, return_counts=True)[1]
    hold_stratify = hold_labels if hold_counts.min() >= 2 else None
    idx_val, idx_test = train_test_split(
        idx_hold,
        test_size=0.50,
        stratify=hold_stratify,
        random_state=seed,
    )

    def gather(idx_list: list[int]) -> tuple[np.ndarray, np.ndarray]:
        x = np.concatenate([file_windows[i] for i in idx_list], axis=0).astype(np.float32)
        y = np.concatenate(
            [np.full(len(file_windows[i]), file_labels[i], dtype=np.int64) for i in idx_list]
        )
        return x, y

    return {"train": gather(idx_train), "val": gather(idx_val), "test": gather(idx_test)}


def save_split(split: dict[str, tuple[np.ndarray, np.ndarray]], out_root: Path) -> None:
    """Save split arrays as torch tensor dictionaries."""

    out_root.mkdir(parents=True, exist_ok=True)
    for name, (x, y) in split.items():
        torch.save({"x": torch.from_numpy(x), "y": torch.from_numpy(y)}, out_root / f"{name}.pt")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build CWRU split from raw .mat files.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_cmd = subparsers.add_parser("build-split")
    build_cmd.add_argument("--raw-root", type=Path, default=Path("data/raw/cwru"))
    build_cmd.add_argument("--out-root", type=Path, default=Path("data/processed/cwru"))
    build_cmd.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.command == "build-split":
        if not args.raw_root.exists():
            print(
                f"WARN: {args.raw_root} does not exist — skip real-CWRU smoke; "
                "document manual fetch in README"
            )
        else:
            split = build_split(args.raw_root, seed=args.seed)
            save_split(split, args.out_root)
            for name, (x, y) in split.items():
                print(f"{name}: {x.shape}, labels={np.bincount(y).tolist()}")
