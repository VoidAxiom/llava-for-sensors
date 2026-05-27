"""Tests for ToyTSEncoder."""
from __future__ import annotations

import pytest
import torch
from torch import nn

from models.encoder import ToyTSEncoder


@pytest.fixture()
def enc() -> ToyTSEncoder:
    torch.manual_seed(0)
    return ToyTSEncoder()


@pytest.fixture()
def sample_input() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(2, 2048)


def test_encoder_output_shape(enc: ToyTSEncoder, sample_input: torch.Tensor) -> None:
    out = enc(sample_input)
    assert out.shape == (2, 32, 512), f"Expected (2, 32, 512), got {out.shape}"


def test_encoder_param_budget(enc: ToyTSEncoder) -> None:
    n_params = sum(p.numel() for p in enc.parameters())
    assert 2_000_000 < n_params < 10_000_000, f"Encoder has {n_params} params, outside PatchTST budget"


def test_encoder_no_nan(enc: ToyTSEncoder, sample_input: torch.Tensor) -> None:
    out = enc(sample_input)
    assert not torch.isnan(out).any(), "NaN in encoder output"
    assert not torch.isinf(out).any(), "Inf in encoder output"


def test_encoder_backward(enc: ToyTSEncoder, sample_input: torch.Tensor) -> None:
    out = enc(sample_input)
    out.sum().backward()
    for name, p in enc.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"


def test_patchtst_uses_attention(enc: ToyTSEncoder) -> None:
    has_attention = any(
        isinstance(module, (nn.TransformerEncoderLayer, nn.MultiheadAttention)) for module in enc.modules()
    )
    assert has_attention, "PatchTST encoder must use transformer attention"


def test_patchtst_param_count(enc: ToyTSEncoder) -> None:
    n_params = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    assert 2_000_000 < n_params < 10_000_000, f"PatchTST has {n_params} trainable params"


def test_patchtst_gradients_flow(enc: ToyTSEncoder) -> None:
    torch.manual_seed(2)
    batch = torch.randn(2, 2048)
    out = enc(batch)
    out.sum().backward()
    for name, p in enc.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"
