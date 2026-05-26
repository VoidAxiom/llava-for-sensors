"""Minimal end-to-end training loop for llava-for-sensors."""

from __future__ import annotations

import json
import pathlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TextIO

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader


@dataclass
class TrainResult:
    final_val_f1: float
    best_val_f1: float
    best_epoch: int
    loss_per_step: list[float]
    val_f1_per_epoch: list[float]
    peak_memory_bytes: int | None
    ckpt_path: pathlib.Path | None


Batch = tuple[torch.Tensor, torch.Tensor, list[str], torch.Tensor]
Sample = tuple[torch.Tensor, torch.Tensor, str, int]


def train_one_run(
    model: nn.Module,
    train_ds: torch.utils.data.Dataset,
    val_ds: torch.utils.data.Dataset,
    *,
    run_id: str,
    n_epochs: int = 5,
    batch_size: int = 2,
    grad_accum: int = 8,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    seed: int = 0,
    log_dir: pathlib.Path | str = "logs",
    ckpt_dir: pathlib.Path | str = "checkpoints",
    device: torch.device | str | None = None,
) -> TrainResult:
    torch.manual_seed(seed)
    np.random.seed(seed)

    if n_epochs <= 0:
        raise ValueError(f"n_epochs must be positive; got {n_epochs}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive; got {batch_size}")
    if grad_accum <= 0:
        raise ValueError(f"grad_accum must be positive; got {grad_accum}")

    resolved_device = _resolve_device(device)
    model.to(resolved_device)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_batch,
    )
    steps_per_epoch = len(train_loader)
    if steps_per_epoch == 0:
        raise ValueError("train_ds must produce at least one batch")

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("model must have at least one trainable parameter")

    optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler_t_max = float(n_epochs * steps_per_epoch / grad_accum)
    scheduler = CosineAnnealingLR(optimizer, T_max=scheduler_t_max)
    loss_fn = nn.CrossEntropyLoss()

    log_path = pathlib.Path(log_dir) / f"{run_id}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_run_dir = pathlib.Path(ckpt_dir) / run_id
    ckpt_run_dir.mkdir(parents=True, exist_ok=True)

    loss_per_step: list[float] = []
    val_f1_per_epoch: list[float] = []
    best_val_f1 = float("-inf")
    best_epoch = -1
    best_ckpt_path: pathlib.Path | None = None
    peak_memory_bytes: int | None = None
    global_step = 0

    with log_path.open("w", encoding="utf-8") as log_file:
        for epoch_idx in range(n_epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            epoch_peak_memory: int | None = None

            for batch_idx, (sensor, image, text, label) in enumerate(train_loader):
                sensor = sensor.to(resolved_device)
                image = image.to(resolved_device)
                label = label.to(resolved_device)

                logits = model.forward(sensor, image, text)
                loss = loss_fn(logits, label)
                scaled_loss = loss / grad_accum
                scaled_loss.backward()

                is_accum_boundary = (batch_idx + 1) % grad_accum == 0
                is_last_batch = (batch_idx + 1) == steps_per_epoch
                if is_accum_boundary or is_last_batch:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                memory_bytes = _memory_bytes(resolved_device)
                if memory_bytes is not None:
                    epoch_peak_memory = _max_optional(epoch_peak_memory, memory_bytes)
                    peak_memory_bytes = _max_optional(peak_memory_bytes, memory_bytes)

                global_step += 1
                scaled_loss_float = float(scaled_loss.detach().cpu().item())
                loss_per_step.append(scaled_loss_float)
                _write_jsonl(
                    log_file,
                    {
                        "event": "step",
                        "step": global_step,
                        "epoch": epoch_idx,
                        "loss": scaled_loss_float,
                        "lr": float(optimizer.param_groups[0]["lr"]),
                        "memory_bytes": memory_bytes,
                    },
                )

            val_f1 = _evaluate_macro_f1(model, val_loader, resolved_device)
            val_f1_per_epoch.append(val_f1)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch_idx
                best_ckpt_path = _save_best_checkpoint(model, ckpt_run_dir)

            _write_jsonl(
                log_file,
                {
                    "event": "epoch",
                    "epoch": epoch_idx,
                    "val_f1": val_f1,
                    "best_f1": best_val_f1,
                    "memory_peak_bytes": epoch_peak_memory,
                },
            )

    return TrainResult(
        final_val_f1=val_f1_per_epoch[-1],
        best_val_f1=best_val_f1,
        best_epoch=best_epoch,
        loss_per_step=loss_per_step,
        val_f1_per_epoch=val_f1_per_epoch,
        peak_memory_bytes=peak_memory_bytes,
        ckpt_path=best_ckpt_path,
    )


def _resolve_device(device: torch.device | str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _collate_batch(batch: Sequence[Sample]) -> Batch:
    sensors: list[torch.Tensor] = []
    images: list[torch.Tensor] = []
    texts: list[str] = []
    labels: list[int] = []
    for sensor, image, text, label in batch:
        sensors.append(sensor)
        images.append(image)
        texts.append(text)
        labels.append(label)
    return (
        torch.stack(sensors),
        torch.stack(images),
        texts,
        torch.tensor(labels, dtype=torch.long),
    )


def _evaluate_macro_f1(
    model: nn.Module,
    val_loader: DataLoader[Batch],
    device: torch.device,
) -> float:
    model.eval()
    preds: list[int] = []
    targets: list[int] = []
    with torch.no_grad():
        for sensor, image, text, label in val_loader:
            sensor = sensor.to(device)
            image = image.to(device)
            label = label.to(device)
            logits = model.forward(sensor, image, text)
            batch_preds = torch.argmax(logits, dim=1).detach().cpu().tolist()
            batch_targets = label.detach().cpu().tolist()
            preds.extend(int(pred) for pred in batch_preds)
            targets.extend(int(target) for target in batch_targets)
    return _macro_f1(preds, targets)


def _macro_f1(preds: Sequence[int], targets: Sequence[int]) -> float:
    pred_array = np.asarray(preds, dtype=np.int64)
    target_array = np.asarray(targets, dtype=np.int64)
    if pred_array.size == 0 or target_array.size == 0:
        return 0.0

    labels = np.union1d(pred_array, target_array)
    scores: list[float] = []
    for label in labels:
        pred_match = pred_array == label
        target_match = target_array == label
        true_positive = int(np.sum(pred_match & target_match))
        false_positive = int(np.sum(pred_match & ~target_match))
        false_negative = int(np.sum(~pred_match & target_match))

        precision_den = true_positive + false_positive
        recall_den = true_positive + false_negative
        precision = true_positive / precision_den if precision_den else 0.0
        recall = true_positive / recall_den if recall_den else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        scores.append(f1)

    return float(np.mean(scores))


def _memory_bytes(device: torch.device) -> int | None:
    if device.type != "mps":
        return None
    return int(torch.mps.driver_allocated_memory())


def _max_optional(current: int | None, value: int) -> int:
    if current is None:
        return value
    return max(current, value)


def _save_best_checkpoint(model: nn.Module, ckpt_run_dir: pathlib.Path) -> pathlib.Path:
    save_pretrained = getattr(model, "save_pretrained", None)
    if callable(save_pretrained):
        ckpt_path = ckpt_run_dir / "best"
        save_pretrained(str(ckpt_path))
        return ckpt_path

    ckpt_path = ckpt_run_dir / "best.pt"
    torch.save(model.state_dict(), ckpt_path)
    return ckpt_path


def _write_jsonl(log_file: TextIO, payload: dict[str, object]) -> None:
    log_file.write(json.dumps(payload) + "\n")
    log_file.flush()
