"""Torch dataset wrapper for the synthetic cross-modal toy data."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import torch
from PIL import Image as PILImage
from torch.utils.data import Dataset

from data.cwru import build_split
from data.images import get_image_for_label
from data.notes import synthesize_note
from data.synthetic import SyntheticSample, generate

_FIXTURE_ROOT = Path(__file__).parent / "test_assets" / "cwru"
_RAW_ROOT = Path("data/raw/cwru")


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
    data/test_assets/cwru/ fixture path for CI.
    mode='synthetic': delegates to an inner ToyDataset (legacy; image is Tensor).
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
            raw_root = _RAW_ROOT if _RAW_ROOT.exists() else _FIXTURE_ROOT
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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, object, str, int]:
        """Return (sensor, image, text, label).

        CWRU mode: image is PIL.Image.Image (224x224 RGB).
        Synthetic mode: image is torch.Tensor (uint8, legacy ToyDataset format).
        """
        if self._mode == "cwru":
            sensor = torch.tensor(self._x[idx], dtype=torch.float32)
            label = int(self._y[idx])
            image: PILImage.Image = get_image_for_label(label)
            text: str = synthesize_note(label, load_hp=1, fault_diameter_in=0.007)
            return sensor, image, text, label
        return self._toy[idx]
