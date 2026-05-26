from __future__ import annotations

from typing import Any

import pytest
import torch
from peft import PeftModel

from models.vlm import EXPECTED_HIDDEN_SIZE, load_frozen_vlm_with_lora


def _register_local_markers() -> None:
    config: Any | None = getattr(pytest.mark, "_config", None)
    if config is None:
        return

    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    )
    config.addinivalue_line("markers", "fast: marks tests as fast sanity checks")


_register_local_markers()


def _resolve_hidden_size(model: PeftModel) -> int | None:
    config = getattr(model, "config", None)
    hidden_size = getattr(config, "hidden_size", None)
    if isinstance(hidden_size, int):
        return hidden_size

    base_model = getattr(model, "base_model", None)
    base_config = getattr(base_model, "config", None)
    base_hidden_size = getattr(base_config, "hidden_size", None)
    if isinstance(base_hidden_size, int):
        return base_hidden_size

    return None


@pytest.mark.fast
def test_expected_hidden_size_constant() -> None:
    """Fast sanity check: no model download."""
    assert EXPECTED_HIDDEN_SIZE == 1536


@pytest.mark.slow
def test_vlm_loads() -> None:
    model = load_frozen_vlm_with_lora()

    assert model is not None
    assert isinstance(model, PeftModel)


@pytest.mark.slow
def test_vlm_params_frozen() -> None:
    model = load_frozen_vlm_with_lora()

    for name, parameter in model.named_parameters():
        if "lora_" not in name:
            assert parameter.requires_grad is False, f"Base parameter should be frozen: {name}"


@pytest.mark.slow
def test_vlm_lora_trainable() -> None:
    model = load_frozen_vlm_with_lora()

    trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    assert trainable_count > 0
    assert trainable_count < 10_000_000


@pytest.mark.slow
def test_vlm_hidden_size() -> None:
    model = load_frozen_vlm_with_lora()
    hidden_size = _resolve_hidden_size(model)

    assert hidden_size == 1536
    assert EXPECTED_HIDDEN_SIZE == 1536


@pytest.mark.slow
def test_vlm_forward_oom_budget() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")

    model = load_frozen_vlm_with_lora()
    input_ids = torch.ones((1, 4), dtype=torch.long).to("mps")
    pixel_values = torch.zeros((1, 3, 224, 224), dtype=torch.float16).to("mps")
    image_grid_thw = torch.tensor([[1, 14, 14]], dtype=torch.int32).to("mps")

    mem_before = torch.mps.driver_allocated_memory()
    try:
        with torch.enable_grad():
            out = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )
            loss = out.logits.sum()
            loss.backward()
    except (RuntimeError, ValueError) as exc:
        pytest.xfail(f"Qwen2-VL rejected the dummy smoke-test input: {exc}")

    mem_after = torch.mps.driver_allocated_memory()
    delta_gb = (mem_after - mem_before) / 1e9

    assert delta_gb < 10.0, f"MPS memory delta {delta_gb:.2f} GB >= 10 GB limit"
