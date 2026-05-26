"""Tests for FusionAdapter."""
from __future__ import annotations

import pytest
import torch

from models.fusion import FusionAdapter


@pytest.fixture()
def fus() -> FusionAdapter:
    torch.manual_seed(0)
    return FusionAdapter()


@pytest.fixture()
def enc_out() -> torch.Tensor:
    torch.manual_seed(2)
    return torch.randn(2, 32, 512)


def test_fusion_output_shape(fus: FusionAdapter, enc_out: torch.Tensor) -> None:
    out = fus(enc_out)
    assert out.shape == (2, 16, 1536), f"Expected (2, 16, 1536), got {out.shape}"


def test_fusion_no_nan(fus: FusionAdapter, enc_out: torch.Tensor) -> None:
    out = fus(enc_out)
    assert not torch.isnan(out).any(), "NaN in fusion output"
    assert not torch.isinf(out).any(), "Inf in fusion output"


def test_fusion_backward(fus: FusionAdapter, enc_out: torch.Tensor) -> None:
    out = fus(enc_out)
    out.sum().backward()
    for name, p in fus.named_parameters():
        assert p.grad is not None, f"No gradient for parameter {name}"


@pytest.mark.slow
def test_fusion_gradcheck() -> None:
    """Numerical gradient check on a tiny double-precision instance."""
    torch.manual_seed(42)
    # Use a tiny adapter to keep gradcheck fast (full 1536-dim is too slow)
    fus_tiny = FusionAdapter(d_enc=4, d_vlm=8, t_sensor=2, n_heads=2).double()
    x = torch.randn(1, 32, 4, dtype=torch.float64, requires_grad=True)
    result = torch.autograd.gradcheck(
        fus_tiny,
        (x,),
        eps=1e-4,
        atol=1e-3,
        rtol=1e-3,
        raise_exception=True,
        fast_mode=True,
    )
    assert result, "gradcheck failed for FusionAdapter"
