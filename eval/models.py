"""Ablation model variants for the toy multimodal experiment."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
from PIL import Image
from torch import Tensor

from models.encoder import ToyTSEncoder
from models.fusion import D_VLM, T_SENSOR_TOKENS, FusionAdapter
from models.vlm import VLMProcessor, load_frozen_vlm_with_lora

_LOGGER = logging.getLogger(__name__)


class ClassificationHead(nn.Module):
    """Linear classifier over a pooled representation."""

    def __init__(self, hidden_size: int, n_classes: int = 4) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.proj = nn.Linear(hidden_size, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = x.to(dtype=self.proj.weight.dtype)
        return self.proj(self.norm(x))


class SensorsOnlyModel(nn.Module):
    """Sensor encoder plus classifier for the sensors-only ablation."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = ToyTSEncoder()
        self.head = ClassificationHead(hidden_size=512)

    def forward(self, sensor: Tensor, image: Tensor, text: list[str]) -> Tensor:
        del image, text
        encoded = self.encoder(sensor)
        pooled = encoded.mean(dim=1)
        return self.head(pooled)


class VisionTextModel(nn.Module):
    """Frozen VLM plus classifier for the vision+text ablation."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
        _vlm: nn.Module | None = None,
        _processor: Any | None = None,
    ) -> None:
        super().__init__()
        self.vlm = _vlm if _vlm is not None else load_frozen_vlm_with_lora(model_id)
        self.processor = _processor if _processor is not None else _load_qwen_processor(model_id)
        self.head = ClassificationHead(hidden_size=D_VLM)
        self.head.to(_module_device(self.vlm, torch.device("cpu")))

    def forward(self, sensor: Tensor, image: Tensor, text: list[str]) -> Tensor:
        del sensor
        device = _module_device(self.vlm, image.device)
        inputs = _prepare_processor_inputs(self.processor, image, text, device)
        output = self.vlm(**inputs, output_hidden_states=True)
        mask = inputs.get("attention_mask")
        pooled = _pool_vlm_output(output, mask if isinstance(mask, Tensor) else None)
        return self.head(pooled)


class AllThreeModel(nn.Module):
    """Sensor-token fusion with the frozen VLM for the all-three ablation."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
        _vlm: nn.Module | None = None,
        _processor: Any | None = None,
    ) -> None:
        super().__init__()
        self.encoder = ToyTSEncoder()
        self.fusion = FusionAdapter()
        self.vlm = _vlm if _vlm is not None else load_frozen_vlm_with_lora(model_id)
        self.processor = _processor if _processor is not None else _load_qwen_processor(model_id)
        self.head = ClassificationHead(hidden_size=D_VLM)
        device = _module_device(self.vlm, torch.device("cpu"))
        self.encoder.to(device)
        self.fusion.to(device)
        self.head.to(device)

    def forward(self, sensor: Tensor, image: Tensor, text: list[str]) -> Tensor:
        device = _module_device(self.vlm, sensor.device)
        sensor = sensor.to(device)
        inputs = _prepare_processor_inputs(self.processor, image, text, device)

        input_ids = inputs.pop("input_ids", None)
        if not isinstance(input_ids, Tensor):
            raise ValueError("processor output must include tensor input_ids")
        inputs["input_ids"] = input_ids

        attention_mask = inputs.pop("attention_mask", None)
        if attention_mask is not None and not isinstance(attention_mask, Tensor):
            raise ValueError("processor attention_mask must be a tensor when provided")

        sensor_tokens = self.fusion(self.encoder(sensor))
        text_image_embeds = self.vlm.get_input_embeddings()(input_ids)
        sensor_tokens = sensor_tokens.to(device=device, dtype=text_image_embeds.dtype)
        combined_embeds = torch.cat([sensor_tokens, text_image_embeds], dim=1)

        if attention_mask is not None:
            batch_size = combined_embeds.shape[0]
            sensor_mask = torch.ones(
                batch_size,
                T_SENSOR_TOKENS,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            inputs["attention_mask"] = torch.cat([sensor_mask, attention_mask], dim=1)

        output = self.vlm(
            inputs_embeds=combined_embeds,
            output_hidden_states=True,
            **inputs,
        )
        mask = inputs.get("attention_mask")
        pooled = _pool_vlm_output(output, mask if isinstance(mask, Tensor) else None)
        return self.head(pooled)


class _TextOnlyProcessor:
    """Fallback processor that tokenizes text only (no image/video pipeline).

    Used when torchvision is not available.
    """

    def __init__(self, model_id: str) -> None:
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")

    def __call__(
        self,
        text: list[str] | None = None,
        images: list[object] | None = None,
        return_tensors: str | None = None,
        padding: bool = True,
        **kw: object,
    ) -> dict[str, Tensor]:
        del images, return_tensors, kw
        if text is None:
            text = [""]
        encoded = self._tokenizer(
            text,
            return_tensors="pt",
            padding=padding,
            truncation=True,
            max_length=64,
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }


def _load_qwen_processor(model_id: str) -> object:
    try:
        return VLMProcessor.from_pretrained(model_id)
    except ImportError:
        _LOGGER.warning(
            "AutoProcessor unavailable (torchvision missing); using text-only fallback"
        )
        return _TextOnlyProcessor(model_id)


def _build_qwen_chat_prompt(text: str) -> str:
    """Wrap a text description in a minimal Qwen2-VL single-image chat prompt."""
    return (
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>"
        f"{text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _prepare_processor_inputs(
    processor: Any,
    image: Tensor,
    text: list[str],
    device: torch.device,
) -> dict[str, Any]:
    pil_images = _tensor_batch_to_pil(image)
    if not isinstance(processor, _TextOnlyProcessor):
        text = [_build_qwen_chat_prompt(item) for item in text]
    raw_inputs = processor(text=text, images=pil_images, return_tensors="pt", padding=True)
    return _move_batch_to_device(raw_inputs, device)


def _tensor_batch_to_pil(image: Tensor) -> list[Image.Image]:
    images: list[Image.Image] = []
    for index in range(image.shape[0]):
        array = image[index].detach().cpu().numpy()
        images.append(Image.fromarray(array))
    return images


def _move_batch_to_device(inputs: Any, device: torch.device) -> dict[str, Any]:
    to_device = getattr(inputs, "to", None)
    if callable(to_device):
        moved = to_device(device)
        if isinstance(moved, Mapping):
            return dict(moved)

    if not isinstance(inputs, Mapping):
        raise TypeError("processor output must be a mapping or BatchEncoding")

    return {
        key: value.to(device) if isinstance(value, Tensor) else value
        for key, value in inputs.items()
    }


def _module_device(module: nn.Module, fallback: torch.device) -> torch.device:
    parameter = next(module.parameters(), None)
    if parameter is None:
        return fallback
    return parameter.device


def _pool_vlm_output(output: Any, attention_mask: Tensor | None = None) -> Tensor:
    hidden_states = getattr(output, "hidden_states", None)
    if hidden_states is not None and len(hidden_states) > 0:
        hidden_state = hidden_states[-1]
        if isinstance(hidden_state, Tensor):
            return _masked_mean(hidden_state, attention_mask)

    last_hidden_state = getattr(output, "last_hidden_state", None)
    if isinstance(last_hidden_state, Tensor):
        return _masked_mean(last_hidden_state, attention_mask)

    logits = getattr(output, "logits", None)
    if isinstance(logits, Tensor):
        return logits[:, -1, :D_VLM]

    raise ValueError("VLM output must include hidden states, last_hidden_state, or logits")


def _masked_mean(hidden_state: Tensor, attention_mask: Tensor | None) -> Tensor:
    """Mean-pool hidden states over real (non-pad) token positions."""
    if attention_mask is None:
        return hidden_state.mean(dim=1)

    mask = attention_mask.to(dtype=hidden_state.dtype, device=hidden_state.device)
    mask_expanded = mask.unsqueeze(-1)
    sum_hidden = (hidden_state * mask_expanded).sum(dim=1)
    count = mask_expanded.sum(dim=1).clamp(min=1.0)
    return sum_hidden / count
