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


def _resolve_hidden_size(model: Any) -> int | None:
    """Walk PEFT + transformers wrappers to find config.hidden_size.

    PEFT wraps with .base_model; transformers wraps decoder with .model.
    For Qwen2VLForConditionalGeneration, the LM config sits at
    base_model.model.config (PEFT) or model.model.config (no PEFT).
    """
    candidates = []
    obj: Any = model
    for _ in range(4):
        candidates.append(obj)
        nxt = getattr(obj, "base_model", None) or getattr(obj, "model", None)
        if nxt is None or nxt is obj:
            break
        obj = nxt

    for c in candidates:
        cfg = getattr(c, "config", None)
        if cfg is not None:
            hs = getattr(cfg, "hidden_size", None)
            if isinstance(hs, int):
                return hs
            for sub_attr in ("text_config", "llm_config", "language_config"):
                sub = getattr(cfg, sub_attr, None)
                if sub is not None:
                    hs = getattr(sub, "hidden_size", None)
                    if isinstance(hs, int):
                        return hs
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
    from transformers import AutoTokenizer
    torch.mps.empty_cache()
    model = load_frozen_vlm_with_lora()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    # Text-only smoke: forces a real forward+backward through the LoRA-wrapped LLM
    # without needing torchvision (the AutoProcessor's video pipeline). The OOM
    # budget is about unified memory under LoRA-only training, not about input
    # modality.
    inputs = tokenizer("hello", return_tensors="pt").to("mps")
    before = torch.mps.driver_allocated_memory()
    with torch.enable_grad():
        out = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        loss = out.logits.float().sum()
        loss.backward()
    after = torch.mps.driver_allocated_memory()
    delta_gb = (after - before) / 1024**3
    print(f"OOM smoke (text-only): delta = {delta_gb:.2f} GB")
    assert delta_gb < 10.0, f"Forward+backward exceeded 10 GB budget: {delta_gb:.2f} GB"
