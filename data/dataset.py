"""Torch dataset wrapper for the synthetic cross-modal toy data."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.utils.data import Dataset

from data.synthetic import SyntheticSample, generate


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
