from __future__ import annotations

import csv
import pathlib

import pytest
import torch
import torch.nn as nn

import eval.ablation as ablation
import scripts.run_full_ablation as orchestrator
import train.loop as train_loop


def test_dry_run_prints_full_schedule_no_training(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out_csv = tmp_path / "dry.csv"

    def fail_run_single(
        condition: str,
        seed: int,
        mode: str = "synthetic",
        n_epochs: int = 5,
        samples_per_class: int = 250,
        device: str | None = None,
    ) -> dict:
        del condition, seed, mode, n_epochs, samples_per_class, device
        raise AssertionError("dry-run must not train")

    monkeypatch.setattr(orchestrator, "run_single", fail_run_single)

    exit_code = orchestrator._main(
        ["--dry-run", "--seeds", "2", "--out-csv", str(out_csv)],
    )

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert stdout.splitlines() == [
        "PLAN: condition=sensors-only seed=0",
        "PLAN: condition=sensors-only seed=1",
        "PLAN: condition=vision+text seed=0",
        "PLAN: condition=vision+text seed=1",
        "PLAN: condition=all-three seed=0",
        "PLAN: condition=all-three seed=1",
    ]
    assert "DONE:" not in stdout
    assert "DONE " not in stdout
    assert not out_csv.exists()


def test_resume_skips_existing_rows(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out_csv = tmp_path / "results.csv"
    _write_existing_rows(out_csv)
    _patch_model_builder(monkeypatch)
    _patch_train_one_run(monkeypatch)

    exit_code = orchestrator._main(["--seeds", "1", "--epochs", "1", "--out-csv", str(out_csv)])

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert [
        line for line in stdout.splitlines() if line.startswith("SKIP condition=")
    ] == [
        "SKIP condition=sensors-only seed=0 (already in CSV)",
        "SKIP condition=vision+text seed=0 (already in CSV)",
    ]

    with out_csv.open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert len(rows) == 3
    assert rows[-1]["condition"] == "all-three"
    assert rows[-1]["seed"] == "0"


def test_csv_header_is_extended_schema(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_csv = tmp_path / "fresh.csv"
    _patch_model_builder(monkeypatch)
    _patch_train_one_run(monkeypatch)

    exit_code = orchestrator._main(["--seeds", "1", "--epochs", "1", "--out-csv", str(out_csv)])

    with out_csv.open("r", encoding="utf-8", newline="") as csv_file:
        header = next(csv.reader(csv_file))

    assert exit_code == 0
    assert header == ["condition", "seed", "final_val_f1", "wall_time_s", "peak_memory_bytes"]


def test_appends_one_row_per_run_atomically(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_csv = tmp_path / "atomic.csv"
    row_counts_after_current_run: list[int] = []
    _patch_model_builder(monkeypatch)

    def recording_train_one_run(
        model: nn.Module,
        train_ds: torch.utils.data.Dataset,
        val_ds: torch.utils.data.Dataset,
        **kwargs: object,
    ) -> train_loop.TrainResult:
        del model, train_ds, val_ds, kwargs
        row_counts_after_current_run.append(_csv_row_count(out_csv) + 1)
        return _fake_train_result()

    monkeypatch.setattr(train_loop, "train_one_run", recording_train_one_run)

    exit_code = orchestrator._main(["--seeds", "1", "--epochs", "1", "--out-csv", str(out_csv)])

    assert exit_code == 0
    assert row_counts_after_current_run == [1, 2, 3]
    assert _csv_row_count(out_csv) == 3


def _patch_model_builder(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_build_model(condition: str) -> nn.Module:
        del condition
        return nn.Linear(1, 1)

    monkeypatch.setattr(ablation, "_build_model", fake_build_model)


def _patch_train_one_run(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_train_one_run(
        model: nn.Module,
        train_ds: torch.utils.data.Dataset,
        val_ds: torch.utils.data.Dataset,
        **kwargs: object,
    ) -> train_loop.TrainResult:
        del model, train_ds, val_ds, kwargs
        return _fake_train_result()

    monkeypatch.setattr(train_loop, "train_one_run", fake_train_one_run)


def _fake_train_result() -> train_loop.TrainResult:
    return train_loop.TrainResult(
        final_val_f1=0.5,
        best_val_f1=0.5,
        best_epoch=0,
        loss_per_step=[0.1],
        val_f1_per_epoch=[0.5],
        peak_memory_bytes=None,
        ckpt_path=None,
    )


def _write_existing_rows(csv_path: pathlib.Path) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["condition", "seed", "final_val_f1", "wall_time_s", "peak_memory_bytes"])
        writer.writerow(["sensors-only", 0, 0.4, 1.0, ""])
        writer.writerow(["vision+text", 0, 0.45, 1.1, ""])


def _csv_row_count(csv_path: pathlib.Path) -> int:
    if not csv_path.exists():
        return 0
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        return sum(1 for _ in csv.DictReader(csv_file))
