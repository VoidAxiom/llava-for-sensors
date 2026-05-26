"""Run the three-condition toy ablation experiment."""

from __future__ import annotations

import argparse
import csv
import pathlib
import time
from typing import Optional

import torch

from data.dataset import ToyDataset
from data.synthetic import N_CLASSES, SyntheticSample, generate
from eval.models import AllThreeModel, SensorsOnlyModel, VisionTextModel
from train.loop import train_one_run


_CONDITIONS = ["sensors-only", "vision+text", "all-three"]


def run_ablation(
    n_seeds: int = 5,
    n_epochs: int = 5,
    out_csv: str = "eval/results_toy.csv",
    samples_per_class: int = 250,
    device: Optional[str] = None,
) -> pathlib.Path:
    samples = generate(n=samples_per_class * N_CLASSES, seed=0)
    train_samples, val_samples = _stratified_split(samples)
    train_ds = ToyDataset(train_samples)
    val_ds = ToyDataset(val_samples)

    rows: list[tuple[str, int, float, float]] = []
    for seed in range(n_seeds):
        for condition in _CONDITIONS:
            torch.manual_seed(seed)
            model = _build_model(condition)
            start = time.time()
            result = train_one_run(
                model,
                train_ds,
                val_ds,
                run_id=f"{condition}-seed{seed}",
                n_epochs=n_epochs,
                seed=seed,
                device=device,
            )
            rows.append((condition, seed, result.final_val_f1, time.time() - start))

    output_path = pathlib.Path(out_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["condition", "seed", "final_val_f1", "wall_time_s"])
        writer.writerows(rows)

    return output_path


def _build_model(condition: str) -> torch.nn.Module:
    if condition == "sensors-only":
        return SensorsOnlyModel()
    if condition == "vision+text":
        return VisionTextModel()
    if condition == "all-three":
        return AllThreeModel()
    raise ValueError(f"unknown condition: {condition}")


def _stratified_split(
    samples: list[SyntheticSample],
) -> tuple[list[SyntheticSample], list[SyntheticSample]]:
    grouped: list[list[SyntheticSample]] = [[] for _ in range(N_CLASSES)]
    for sample in sorted(samples, key=lambda item: int(item["label"])):
        grouped[int(sample["label"])].append(sample)

    train_groups: list[list[SyntheticSample]] = []
    val_groups: list[list[SyntheticSample]] = []
    for class_samples in grouped:
        split_index = int(len(class_samples) * 0.8)
        train_groups.append(class_samples[:split_index])
        val_groups.append(class_samples[split_index:])

    return _interleave(train_groups), _interleave(val_groups)


def _interleave(groups: list[list[SyntheticSample]]) -> list[SyntheticSample]:
    interleaved: list[SyntheticSample] = []
    max_len = max((len(group) for group in groups), default=0)
    for index in range(max_len):
        for group in groups:
            if index < len(group):
                interleaved.append(group[index])
    return interleaved


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run the toy modality ablation.")
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--out", default="eval/results_toy.csv")
    parser.add_argument("--samples-per-class", type=int, default=250)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    run_ablation(
        n_seeds=args.n_seeds,
        n_epochs=args.n_epochs,
        out_csv=args.out,
        samples_per_class=args.samples_per_class,
        device=args.device,
    )


if __name__ == "__main__":
    _main()
