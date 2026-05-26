from __future__ import annotations

import json
import os
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from train.loop import TrainResult, _macro_f1, train_one_run


class _TinyLinearModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(2048, 4)

    def forward(
        self,
        sensor: torch.Tensor,
        image: torch.Tensor | None = None,
        text: list[str] | None = None,
    ) -> torch.Tensor:
        return self.head(sensor)


class _TinyDataset(torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor, str, int]]):
    def __init__(self, n: int = 8) -> None:
        generator = torch.Generator().manual_seed(123)
        self._sensors = [torch.randn(2048, generator=generator) for _ in range(n)]
        self._images = [torch.zeros(224, 224, 3, dtype=torch.uint8) for _ in range(n)]

    def __len__(self) -> int:
        return len(self._sensors)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str, int]:
        return self._sensors[index], self._images[index], "text", 0


def test_train_result_dataclass_fields() -> None:
    field_names = [field.name for field in fields(TrainResult)]

    assert field_names == [
        "final_val_f1",
        "best_val_f1",
        "best_epoch",
        "loss_per_step",
        "val_f1_per_epoch",
        "peak_memory_bytes",
        "ckpt_path",
    ]


def test_train_one_run_returns_result(tmp_path: Path) -> None:
    model = _TinyLinearModel()
    dataset = _TinyDataset(n=8)

    result = train_one_run(
        model,
        dataset,
        dataset,
        run_id="test_fast",
        n_epochs=1,
        batch_size=4,
        grad_accum=2,
        device="cpu",
        log_dir=tmp_path / "logs",
        ckpt_dir=tmp_path / "ckpt",
    )

    assert isinstance(result, TrainResult)
    assert len(result.loss_per_step) > 0


@pytest.mark.slow
def test_loss_decreases_on_toy_smoke(tmp_path: Path) -> None:
    if not os.environ.get("RUN_SLOW_TESTS"):
        pytest.skip("set RUN_SLOW_TESTS=1")

    from data.dataset import ToyDataset
    from data.synthetic import generate

    samples = generate(n=80, seed=0)
    ds_train = ToyDataset(samples[:60])
    ds_val = ToyDataset(samples[60:])
    model = _TinyLinearModel()

    result = train_one_run(
        model,
        ds_train,
        ds_val,
        run_id="smoke",
        n_epochs=2,
        batch_size=4,
        grad_accum=2,
        device="cpu",
        log_dir=str(tmp_path / "logs"),
        ckpt_dir=str(tmp_path / "ckpt"),
    )

    assert np.mean(result.loss_per_step[-5:]) < np.mean(result.loss_per_step[:5])
    assert result.final_val_f1 > 0.25


@pytest.mark.slow
def test_checkpoint_saved_on_improvement(tmp_path: Path) -> None:
    if not os.environ.get("RUN_SLOW_TESTS"):
        pytest.skip("set RUN_SLOW_TESTS=1")

    from data.dataset import ToyDataset
    from data.synthetic import generate

    samples = generate(n=80, seed=0)
    ds_train = ToyDataset(samples[:60])
    ds_val = ToyDataset(samples[60:])
    model = _TinyLinearModel()

    result = train_one_run(
        model,
        ds_train,
        ds_val,
        run_id="ckpt_test",
        n_epochs=2,
        batch_size=4,
        grad_accum=2,
        device="cpu",
        log_dir=str(tmp_path / "logs"),
        ckpt_dir=str(tmp_path / "ckpt"),
    )

    assert result.ckpt_path is not None and result.ckpt_path.exists()


@pytest.mark.slow
def test_jsonl_logging_emits_step_and_epoch_events(tmp_path: Path) -> None:
    if not os.environ.get("RUN_SLOW_TESTS"):
        pytest.skip("set RUN_SLOW_TESTS=1")

    model = _TinyLinearModel()
    dataset = _TinyDataset(n=8)

    train_one_run(
        model,
        dataset,
        dataset,
        run_id="log_test",
        n_epochs=1,
        batch_size=4,
        grad_accum=2,
        device="cpu",
        log_dir=tmp_path / "logs",
        ckpt_dir=tmp_path / "ckpt",
    )

    log_path = tmp_path / "logs" / "log_test.jsonl"
    lines = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    step_line = next(line for line in lines if line["event"] == "step")
    epoch_line = next(line for line in lines if line["event"] == "epoch")

    assert any(line["event"] == "step" for line in lines)
    assert any(line["event"] == "epoch" for line in lines)
    assert {"epoch", "val_f1", "best_f1", "memory_peak_bytes"} <= epoch_line.keys()
    assert {"step", "epoch", "loss", "lr", "memory_bytes"} <= step_line.keys()


@pytest.mark.slow
def test_determinism_same_seed(tmp_path: Path) -> None:
    if not os.environ.get("RUN_SLOW_TESTS"):
        pytest.skip("set RUN_SLOW_TESTS=1")

    dataset = _TinyDataset(n=8)
    torch.manual_seed(0)
    first_model = _TinyLinearModel()
    torch.manual_seed(0)
    second_model = _TinyLinearModel()

    first_result = train_one_run(
        first_model,
        dataset,
        dataset,
        run_id="determinism_1",
        n_epochs=1,
        batch_size=4,
        grad_accum=2,
        seed=42,
        device="cpu",
        log_dir=tmp_path / "logs",
        ckpt_dir=tmp_path / "ckpt",
    )
    second_result = train_one_run(
        second_model,
        dataset,
        dataset,
        run_id="determinism_2",
        n_epochs=1,
        batch_size=4,
        grad_accum=2,
        seed=42,
        device="cpu",
        log_dir=tmp_path / "logs",
        ckpt_dir=tmp_path / "ckpt",
    )

    assert len(first_result.loss_per_step) == len(second_result.loss_per_step)
    assert all(
        abs(first_loss - second_loss) < 1e-5
        for first_loss, second_loss in zip(
            first_result.loss_per_step,
            second_result.loss_per_step,
            strict=True,
        )
    )


def test_grad_accum_steps_optimizer_every_n_batches(tmp_path: Path) -> None:
    model = _TinyLinearModel()
    dataset = _TinyDataset(n=8)

    result = train_one_run(
        model,
        dataset,
        dataset,
        run_id="accum",
        n_epochs=1,
        batch_size=2,
        grad_accum=4,
        device="cpu",
        log_dir=tmp_path / "logs",
        ckpt_dir=tmp_path / "ckpt",
    )

    log_path = tmp_path / "logs" / "accum.jsonl"
    lines = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]

    assert len(result.loss_per_step) == 4
    assert sum(line["event"] == "step" for line in lines) == 4
    assert sum(line["event"] == "epoch" for line in lines) == 1


def test_macro_f1_all_classes_averaged() -> None:
    assert abs(_macro_f1([0, 0], [0, 0]) - 0.25) < 1e-6


def test_scheduler_t_max_is_ceil_not_floor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler_calls = {"t_max": 0, "steps": 0}

    class _CountingScheduler:
        def __init__(self, optimizer: torch.optim.Optimizer, T_max: int) -> None:
            del optimizer
            scheduler_calls["t_max"] = T_max

        def step(self) -> None:
            scheduler_calls["steps"] += 1

    monkeypatch.setattr("train.loop.CosineAnnealingLR", _CountingScheduler)

    model = _TinyLinearModel()
    dataset = _TinyDataset(n=10)

    result = train_one_run(
        model,
        dataset,
        dataset,
        run_id="ceil_t_max",
        n_epochs=1,
        batch_size=2,
        grad_accum=4,
        device="cpu",
        log_dir=tmp_path / "logs",
        ckpt_dir=tmp_path / "ckpt",
    )

    # 5 batches with grad_accum 4 flushes once at batch 4 and once at the final
    # partial batch, so ceil(5 / 4) == 2 scheduler steps.
    assert scheduler_calls["t_max"] == 2
    assert scheduler_calls["steps"] == 2
    assert len(result.loss_per_step) == 5
