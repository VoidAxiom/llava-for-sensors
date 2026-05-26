"""Tests for ToyTSEncoder."""
from __future__ import annotations

import pytest
import torch

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
    assert n_params < 2_000_000, f"Encoder has {n_params} params, exceeds 2M budget"


def test_encoder_no_nan(enc: ToyTSEncoder, sample_input: torch.Tensor) -> None:
    out = enc(sample_input)
    assert not torch.isnan(out).any(), "NaN in encoder output"
    assert not torch.isinf(out).any(), "Inf in encoder output"


def test_encoder_backward(enc: ToyTSEncoder, sample_input: torch.Tensor) -> None:
    out = enc(sample_input)
    out.sum().backward()
    for name, p in enc.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"
