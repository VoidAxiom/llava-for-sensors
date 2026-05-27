"""Torch dataset wrapper for the synthetic cross-modal toy data."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from data.cwru import build_split
from data.images import get_image_for_label
from data.notes import synthesize_note
from data.synthetic import SyntheticSample, generate

_FIXTURE_ROOT = Path(__file__).parent / "test_assets" / "cwru"
_PROCESSED_ROOT = Path("data/processed/cwru")
_RAW_ROOT = Path("data/raw/cwru")
_CWRU_CLASS_NAMES = ("normal", "inner_race", "outer_race", "ball")


def _has_usable_cwru_raw(root: Path) -> bool:
    for class_name in _CWRU_CLASS_NAMES:
        class_dir = root / class_name
        if not class_dir.is_dir() or not any(class_dir.glob("*.mat")):
            return False
    return True


class ToyDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str, int]]):
    """Dataset returning sensor, image, text, and label tuples."""

    def __init__(self, samples_or_seed: Sequence[SyntheticSample] | int = 0, n: int = 1000) -> None:
        if isinstance(samples_or_seed, int):
            self._samples = generate(n=n, seed=samples_or_seed)
        else:
            self._samples = list(samples_or_seed)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str, int]:
        sample = self._samples[index]
        sensor = torch.tensor(sample["sensor"], dtype=torch.float32)
        image = torch.tensor(sample["image"], dtype=torch.uint8)
        text = sample["text"]
        label = int(sample["label"])
        return sensor, image, text, label


class BearingFaultDataset(Dataset):
    """Dataset returning (sensor_window, image, text, label) for bearing fault data.

    mode='cwru': reads from data/raw/cwru/ when present; falls back to
    data/test_assets/cwru/ fixture path for CI. Both modes return image as a
    torch.Tensor (uint8).
    mode='synthetic': delegates to an inner ToyDataset.
    """

    def __init__(
        self,
        *,
        mode: Literal["synthetic", "cwru"] = "synthetic",
        split: Literal["train", "val", "test"] = "train",
        **kwargs: object,
    ) -> None:
        self._mode = mode
        if mode == "cwru":
            processed_path = _PROCESSED_ROOT / f"{split}.pt"
            if processed_path.exists():
                data = torch.load(processed_path, weights_only=True)
                self._x = data["x"].numpy()
                self._y = data["y"].numpy()
            else:
                raw_root = _RAW_ROOT if _has_usable_cwru_raw(_RAW_ROOT) else _FIXTURE_ROOT
                all_splits = build_split(raw_root, seed=0)
                self._x, self._y = all_splits[split]
        elif mode == "synthetic":
            seed = int(kwargs.get("seed", 0))
            n = int(kwargs.get("n", 1000))
            self._toy = ToyDataset(seed, n=n)
        else:
            raise ValueError(f"mode must be 'synthetic' or 'cwru'; got {mode!r}")

    def __len__(self) -> int:
        if self._mode == "cwru":
            return len(self._x)
        return len(self._toy)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str, int]:
        """Return (sensor, image, text, label).

        Both modes return image as torch.Tensor (uint8, shape 224x224x3).
        """
        if self._mode == "cwru":
            sensor = torch.tensor(self._x[idx], dtype=torch.float32)
            label = int(self._y[idx])
            image = torch.tensor(np.array(get_image_for_label(label)), dtype=torch.uint8)
            text: str = synthesize_note(label, load_hp=1, fault_diameter_in=0.007)
            return sensor, image, text, label
        return self._toy[idx]
