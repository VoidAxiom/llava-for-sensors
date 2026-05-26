from __future__ import annotations

import csv
import os
import pathlib
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn

from eval.ablation import run_ablation
from eval.headline import PhaseAcceptanceError, run_headline_from_csv
from eval.models import AllThreeModel, SensorsOnlyModel, VisionTextModel


def _register_local_markers() -> None:
    config: Any | None = getattr(pytest.mark, "_config", None)
    if config is None:
        return

    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    )


_register_local_markers()


SKIP_SLOW: bool = os.environ.get("RUN_SLOW_TESTS", "").lower() not in ("1", "true", "yes")
skip_unless_slow = pytest.mark.skipif(SKIP_SLOW, reason="Slow test: set RUN_SLOW_TESTS=1 to enable")


class _StubEmbedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1, 1536))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        return self.anchor.expand(batch_size, seq_len, -1)


class _StubVLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = _StubEmbedding()

    def get_input_embeddings(self) -> nn.Module:
        return self.embed

    def forward(
        self,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        **kwargs: object,
    ) -> SimpleNamespace:
        del attention_mask, output_hidden_states, kwargs
        if inputs_embeds is not None:
            batch_size, seq_len = inputs_embeds.shape[:2]
        elif input_ids is not None:
            batch_size, seq_len = input_ids.shape
        else:
            batch_size, seq_len = 1, 8
        hidden = torch.zeros(batch_size, seq_len, 1536, dtype=torch.float32)
        return SimpleNamespace(
            hidden_states=(hidden,) * 4,
            last_hidden_state=hidden,
            logits=torch.zeros(batch_size, seq_len, 152000, dtype=torch.float32),
        )


class _StubProcessor:
    def __call__(
        self,
        text: list[str] | None = None,
        images: list[object] | None = None,
        return_tensors: str | None = None,
        padding: bool = True,
        **kw: object,
    ) -> dict[str, torch.Tensor]:
        del images, return_tensors, padding, kw
        batch_size = len(text) if text else 1
        return {
            "input_ids": torch.ones(batch_size, 8, dtype=torch.long),
            "attention_mask": torch.ones(batch_size, 8, dtype=torch.long),
        }


def test_models_have_4class_logits() -> None:
    torch.manual_seed(0)
    sensors_only = SensorsOnlyModel()
    stub_processor = _StubProcessor()

    sensor = torch.zeros(2, 2048)
    image = torch.zeros(2, 224, 224, 3, dtype=torch.uint8)
    text = ["a", "b"]

    assert sensors_only.forward(sensor, image, text).shape == (2, 4)
    assert VisionTextModel(_vlm=_StubVLM(), _processor=stub_processor).forward(
        sensor,
        image,
        text,
    ).shape == (2, 4)
    assert AllThreeModel(_vlm=_StubVLM(), _processor=stub_processor).forward(
        sensor,
        image,
        text,
    ).shape == (2, 4)


@skip_unless_slow
@pytest.mark.slow
def test_run_ablation_writes_csv(tmp_path: pathlib.Path) -> None:
    out_csv = run_ablation(
        n_seeds=1,
        n_epochs=1,
        samples_per_class=20,
        out_csv=str(tmp_path / "results.csv"),
    )

    assert out_csv.exists()
    with out_csv.open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert len(rows) == 3
    assert {row["condition"] for row in rows} == {"sensors-only", "vision+text", "all-three"}
    for row in rows:
        final_val_f1 = float(row["final_val_f1"])
        assert np.isfinite(final_val_f1)
        assert 0.0 <= final_val_f1 <= 1.0


def test_headline_gate_rejects_failing_results(tmp_path: pathlib.Path) -> None:
    failing = np.array(
        [
            [0.45, 0.46, 0.44, 0.47, 0.45],
            [0.80, 0.81, 0.79, 0.82, 0.80],
            [0.50, 0.51, 0.49, 0.52, 0.50],
        ]
    )
    csv_path = _write_fake_csv(tmp_path, failing)

    with pytest.raises(PhaseAcceptanceError):
        run_headline_from_csv(csv_path=str(csv_path), out_svg=str(tmp_path / "out.svg"))


def test_headline_gate_rejects_wrong_seed_count(tmp_path: pathlib.Path) -> None:
    one_seed = np.array([[0.90], [0.50], [0.70]])
    csv_path = _write_fake_csv(tmp_path, one_seed)

    with pytest.raises(ValueError, match="5 seeds"):
        run_headline_from_csv(csv_path=str(csv_path), out_svg=str(tmp_path / "out.svg"))


def test_headline_gate_accepts_passing_results(tmp_path: pathlib.Path) -> None:
    passing = np.array(
        [
            [0.45, 0.46, 0.44, 0.47, 0.45],
            [0.55, 0.56, 0.54, 0.57, 0.55],
            [0.92, 0.93, 0.91, 0.94, 0.92],
        ]
    )
    csv_path = _write_fake_csv(tmp_path, passing)

    result = run_headline_from_csv(csv_path=str(csv_path), out_svg=str(tmp_path / "out.svg"))

    assert result["acceptance"] == "passed"
    assert result["gap_vt"] > 0.15
    assert result["gap_so"] > 0.15


def _write_fake_csv(tmp_path: pathlib.Path, results: np.ndarray) -> pathlib.Path:
    csv_path = tmp_path / "test_results.csv"
    conditions = ["sensors-only", "vision+text", "all-three"]
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["condition", "seed", "final_val_f1", "wall_time_s"])
        for condition_index, condition in enumerate(conditions):
            for seed_index, value in enumerate(results[condition_index]):
                writer.writerow([condition, seed_index, float(value), 0.1])
    return csv_path
