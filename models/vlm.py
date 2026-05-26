"""Load a frozen Qwen2-VL model with LoRA adapters for trainable fusion."""

from __future__ import annotations

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoProcessor as VLMProcessor  # noqa: F401
from transformers import Qwen2VLForConditionalGeneration

EXPECTED_HIDDEN_SIZE: int = 1536


def load_frozen_vlm_with_lora(
    model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
) -> "PeftModel":
    """Load Qwen2-VL frozen on all base parameters with LoRA trainable adapters."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        torch_dtype = torch.float16
    else:
        import warnings

        warnings.warn("MPS not available; using CPU. Expect very slow training.")
        device = torch.device("cpu")
        torch_dtype = torch.float32

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
    )

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)

    model = model.to(device)
    return model
