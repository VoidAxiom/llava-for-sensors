"""CWRU Bearing Data Center loader, preprocessor, and stratified splitter.

Manual fetch: see README.md § "CWRU dataset — manual fetch".
Expected layout under data/raw/cwru/:
    normal/          (4 records: 97.mat 98.mat 99.mat 100.mat)
    inner_race/      (multiple records)
    outer_race/      (multiple records)
    ball/            (multiple records)
"""

from __future__ import annotations

import fnmatch
from pathlib import Path

import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
import torch

CLASS_NAMES: tuple[str, ...] = ("normal", "inner_race", "outer_race", "ball")
CLASS_LABELS: dict[str, int] = {"normal": 0, "inner_race": 1, "outer_race": 2, "ball": 3}
WINDOW_SIZE: int = 2048
SAMPLE_RATE_HZ: int = 12000


def load_cwru_mat(path: Path) -> np.ndarray:
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
    return np.asarray(mat_data[drive_end_key]).reshape(-1).astype(np.float32)


def preprocess_to_windows(samples: np.ndarray, window_size: int = WINDOW_SIZE) -> np.ndarray:
    """Convert a 1-D signal into non-overlapping fixed-size windows."""

    n_windows = len(samples) // window_size
    return samples[: n_windows * window_size].reshape(n_windows, window_size).astype(np.float32)


def load_class_windows(class_dir: Path) -> np.ndarray:
    """Load and window every .mat file in one class directory."""

    if not class_dir.exists():
        raise FileNotFoundError(f"Class directory not found: {class_dir}")
    mat_files = sorted(class_dir.glob("*.mat"))
    if not mat_files:
        raise ValueError(f"No .mat files found in {class_dir}")
    windows = [preprocess_to_windows(load_cwru_mat(mat_file)) for mat_file in mat_files]
    return np.concatenate(windows, axis=0)


def build_split(raw_root: Path, seed: int = 0) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Build deterministic stratified train/val/test splits from raw CWRU .mat files."""

    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    for class_name in CLASS_NAMES:
        windows = load_class_windows(raw_root / class_name)
        label = CLASS_LABELS[class_name]
        x_list.append(windows)
        y_list.append(np.full(len(windows), label, dtype=np.int64))

    x_all = np.concatenate(x_list, axis=0).astype(np.float32)
    y_all = np.concatenate(y_list, axis=0).astype(np.int64)
    x_train, x_hold, y_train, y_hold = train_test_split(
        x_all,
        y_all,
        test_size=0.20,
        stratify=y_all,
        random_state=seed,
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_hold,
        y_hold,
        test_size=0.50,
        stratify=y_hold,
        random_state=seed,
    )
    return {"train": (x_train, y_train), "val": (x_val, y_val), "test": (x_test, y_test)}


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
