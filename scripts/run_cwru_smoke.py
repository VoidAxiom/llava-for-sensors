"""Smoke training run on CWRU bearing data.

Loads the CWRU dataset (real or fixture fallback), runs 1 epoch of training
with SensorsOnlyModel to verify the dataset -> model -> loss pipeline works.
Exits 0 with a WARN message if data/raw/cwru/ is not yet fetched.

Usage:
    uv run python scripts/run_cwru_smoke.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


_RAW_ROOT = Path("data/raw/cwru")


def _make_tensor_dataset(base: Dataset) -> Dataset:
    """Wrap a BearingFaultDataset to convert PIL images to uint8 Tensors for collation."""

    class TensorWrapper(Dataset):
        def __init__(self, inner: Dataset) -> None:
            self._inner = inner

        def __len__(self) -> int:
            return len(self._inner)

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str, int]:
            sensor, image, text, label = self._inner[idx]
            img_array = np.array(image, dtype=np.uint8)
            img_tensor = torch.tensor(img_array, dtype=torch.uint8)
            return sensor, img_tensor, text, label

    return TensorWrapper(base)


def main() -> None:
    if not _RAW_ROOT.exists():
        print(
            "WARN: data/raw/cwru/ empty — smoke training deferred until user fetches CWRU;"
            " document in RUNNING_NOTES.md when done"
        )
        return

    from data.dataset import BearingFaultDataset
    from eval.models import SensorsOnlyModel
    from train.loop import train_one_run

    print("data/raw/cwru/ found — running smoke training on real CWRU data")
    train_ds = _make_tensor_dataset(BearingFaultDataset(mode="cwru", split="train"))
    val_ds = _make_tensor_dataset(BearingFaultDataset(mode="cwru", split="val"))

    model = SensorsOnlyModel()
    t0 = time.perf_counter()
    result = train_one_run(
        model,
        train_ds,
        val_ds,
        run_id="cwru-smoke",
        n_epochs=1,
        batch_size=4,
        grad_accum=4,
        lr=1e-4,
        seed=0,
    )
    elapsed = time.perf_counter() - t0

    n_steps = len(result.loss_per_step)
    first_loss = result.loss_per_step[0] if result.loss_per_step else float("nan")
    last_loss = result.loss_per_step[-1] if result.loss_per_step else float("nan")
    print(f"steps: {n_steps}")
    print(f"loss: first={first_loss:.4f}  last={last_loss:.4f}")
    print(f"val_f1: {result.final_val_f1:.4f}")
    print(f"wall_time: {elapsed:.1f}s")
    if result.peak_memory_bytes is not None:
        print(f"peak_memory_mib: {result.peak_memory_bytes / (1024**2):.1f}")


if __name__ == "__main__":
    main()
