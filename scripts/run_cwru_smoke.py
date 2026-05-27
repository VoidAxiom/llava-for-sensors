"""Smoke training run on CWRU bearing data.

Loads the CWRU dataset (real or fixture fallback), runs 1 epoch of training
with SensorsOnlyModel to verify the dataset -> model -> loss pipeline works.
Exits 0 with a WARN message if data/raw/cwru/ is not yet fetched.

Usage:
    uv run python scripts/run_cwru_smoke.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PROJECT_ROOT_STR = str(_PROJECT_ROOT)
if _PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT_STR)

_CWRU_CLASS_NAMES = ("normal", "inner_race", "outer_race", "ball")
_RAW_ROOT = Path("data/raw/cwru")


def _has_usable_cwru_raw(root: Path) -> bool:
    for class_name in _CWRU_CLASS_NAMES:
        class_dir = root / class_name
        if not class_dir.is_dir() or not any(class_dir.glob("*.mat")):
            return False
    return True


def main() -> None:
    if not _has_usable_cwru_raw(_RAW_ROOT):
        print(
            "WARN: data/raw/cwru/ empty — smoke training deferred until user fetches CWRU;"
            " document in RUNNING_NOTES.md when done"
        )
        return

    from data.dataset import BearingFaultDataset
    from eval.models import SensorsOnlyModel
    from train.loop import train_one_run

    print("data/raw/cwru/ found — running smoke training on real CWRU data")
    train_ds = BearingFaultDataset(mode="cwru", split="train")
    val_ds = BearingFaultDataset(mode="cwru", split="val")

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
