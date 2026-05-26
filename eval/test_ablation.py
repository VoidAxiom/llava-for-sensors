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
from PIL import Image as PILImage

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

IMAGE_PAD_TOKEN_ID = 151655


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


class _StubProcessorWithPixelValues:
    def __call__(
        self,
        text: list[str] | None = None,
        images: list[object] | None = None,
        return_tensors: str | None = None,
        padding: bool = True,
        **kw: object,
    ) -> dict[str, torch.Tensor]:
        del return_tensors, padding, kw
        batch_size = len(text) if text else 1
        pixel_values_list: list[torch.Tensor] = []
        if images:
            for img in images:
                if not isinstance(img, PILImage.Image):
                    raise TypeError("images must contain PIL.Image instances")
                arr = np.array(img, dtype=np.float32) / 255.0
                mean_val = float(arr.mean())
                pv = torch.full((3, 16, 16), mean_val, dtype=torch.float32)
                pixel_values_list.append(pv)
                pixel_values_list.append(pv.clone())
        else:
            for _ in range(batch_size):
                pixel_values_list.append(torch.zeros(3, 16, 16, dtype=torch.float32))
                pixel_values_list.append(torch.zeros(3, 16, 16, dtype=torch.float32))
        input_ids = torch.ones(batch_size, 8, dtype=torch.long)
        input_ids[:, :2] = IMAGE_PAD_TOKEN_ID
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones(batch_size, 8, dtype=torch.long),
            "pixel_values": torch.stack(pixel_values_list),
            "image_grid_thw": torch.tensor([[1, 1, 2]] * batch_size, dtype=torch.long),
        }


class _StubVLMWithPixelValues(nn.Module):
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
        pixel_values: torch.Tensor | None = None,
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
        if pixel_values is not None:
            # Add pixel mean to hidden state so different images produce
            # different outputs -- simulates Qwen2VL visual feature scatter.
            pixel_mean = pixel_values.float().mean()
            hidden = hidden + pixel_mean
            # LayerNorm removes uniform offsets; keep one feature image-dependent.
            hidden[:, :, 0] = hidden[:, :, 0] + pixel_mean
        return SimpleNamespace(
            hidden_states=(hidden,) * 4,
            last_hidden_state=hidden,
            logits=torch.zeros(batch_size, seq_len, 152000, dtype=torch.float32),
        )


class _StubVisualOutput:
    def __init__(self, pooler_output: list[torch.Tensor]) -> None:
        self.pooler_output = pooler_output


class _StubVisual(nn.Module):
    dtype = torch.float32

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor | None = None,
    ) -> _StubVisualOutput:
        patch_means = pixel_values.float().mean(dim=(1, 2, 3))
        image_embeds = patch_means.unsqueeze(-1).expand(-1, 1536)
        if grid_thw is not None:
            n_images = int(grid_thw.shape[0])
        else:
            n_images = 1
        patches_per_image = image_embeds.shape[0] // n_images
        pooler_output = [
            image_embeds[index * patches_per_image : (index + 1) * patches_per_image]
            for index in range(n_images)
        ]
        return _StubVisualOutput(pooler_output=pooler_output)


class _StubConfig:
    image_token_id = IMAGE_PAD_TOKEN_ID


class _StubModelWithVisual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual = _StubVisual()
        self.config = _StubConfig()


class _StubVLMWithVisual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = _StubEmbedding()
        self.model = _StubModelWithVisual()
        self.received_inputs_embeds: list[torch.Tensor] = []
        self.forward_calls: list[dict[str, object]] = []

    def get_input_embeddings(self) -> nn.Module:
        return self.embed

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        logits_to_keep: int | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        self.received_inputs_embeds.append(inputs_embeds.detach().clone())
        self.forward_calls.append(
            {
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "output_hidden_states": output_hidden_states,
                "logits_to_keep": logits_to_keep,
                "position_ids": position_ids,
            }
        )
        hidden = inputs_embeds.to(dtype=torch.float32)
        batch_size, seq_len = hidden.shape[:2]
        return SimpleNamespace(
            hidden_states=(hidden,) * 4,
            last_hidden_state=hidden,
            logits=torch.zeros(batch_size, seq_len, 152000, dtype=torch.float32),
        )


def test_all_three_vision_scatter_applied() -> None:
    """AllThreeModel must scatter visual embeddings before calling the VLM."""
    torch.manual_seed(0)
    stub_vlm = _StubVLMWithVisual()
    stub_processor = _StubProcessorWithPixelValues()
    model = AllThreeModel(_vlm=stub_vlm, _processor=stub_processor)
    model.eval()

    sensor = torch.zeros(1, 2048)
    text = ["describe sensor data"]
    black_image = torch.zeros(1, 224, 224, 3, dtype=torch.uint8)
    white_image = torch.full((1, 224, 224, 3), 255, dtype=torch.uint8)

    with torch.no_grad():
        model.forward(sensor, black_image, text)
        model.forward(sensor, white_image, text)

    assert len(stub_vlm.received_inputs_embeds) == 2
    assert not torch.equal(
        stub_vlm.received_inputs_embeds[0],
        stub_vlm.received_inputs_embeds[1],
    )
    for call in stub_vlm.forward_calls:
        assert "pixel_values" not in call
        assert "input_ids" not in call


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


def test_headline_gate_rejects_no_significance(tmp_path: pathlib.Path) -> None:
    """Gate must reject when verdict is not fusion_wins."""
    from unittest.mock import patch

    import eval.headline as hl_module

    high_variance = np.array(
        [
            [0.30, 0.50, 0.20, 0.60, 0.40],
            [0.40, 0.60, 0.30, 0.70, 0.50],
            [0.80, 0.95, 0.70, 0.85, 0.90],
        ]
    )
    csv_path = _write_fake_csv(tmp_path, high_variance)

    original_compute = hl_module.compute_headline

    def patched_compute(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = original_compute(*args, **kwargs)
        result["verdict"] = "no_significant_difference"
        return result

    with patch.object(hl_module, "compute_headline", patched_compute):
        with pytest.raises(PhaseAcceptanceError, match="verdict"):
            run_headline_from_csv(csv_path=str(csv_path), out_svg=str(tmp_path / "out.svg"))


def test_headline_gate_rejects_wrong_seed_count(tmp_path: pathlib.Path) -> None:
    one_seed = np.array([[0.90], [0.50], [0.70]])
    csv_path = _write_fake_csv(tmp_path, one_seed)

    with pytest.raises(ValueError, match="5 seeds"):
        run_headline_from_csv(csv_path=str(csv_path), out_svg=str(tmp_path / "out.svg"))


def test_headline_gate_rejects_duplicate_seeds(tmp_path: pathlib.Path) -> None:
    csv_path = tmp_path / "duplicate_seeds.csv"
    seeds = [0, 0, 1, 2, 3]
    values_by_condition = {
        "sensors-only": [0.45, 0.46, 0.44, 0.47, 0.45],
        "vision+text": [0.55, 0.56, 0.54, 0.57, 0.55],
        "all-three": [0.92, 0.93, 0.91, 0.94, 0.92],
    }
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["condition", "seed", "final_val_f1"])
        for condition, values in values_by_condition.items():
            for seed, value in zip(seeds, values):
                writer.writerow([condition, seed, value])

    with pytest.raises(PhaseAcceptanceError):
        run_headline_from_csv(csv_path=str(csv_path), out_svg=str(tmp_path / "out.svg"))


def test_headline_gate_rejects_duplicate_seed_ids_per_condition(
    tmp_path: pathlib.Path,
) -> None:
    """Gate must reject when a condition has duplicate seed IDs."""
    conditions = ["sensors-only", "vision+text", "all-three"]
    csv_path = tmp_path / "dup_seeds.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "seed", "final_val_f1", "wall_time_s"])
        for cond in conditions:
            for seed, val in zip([0, 0, 1, 2, 3], [0.5, 0.5, 0.6, 0.7, 0.8]):
                writer.writerow([cond, seed, val, 0.1])

    with pytest.raises(PhaseAcceptanceError, match="Duplicate"):
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
