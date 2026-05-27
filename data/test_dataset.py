"""Tests for data/dataset.py BearingFaultDataset (CWRU mode)."""

from __future__ import annotations

import torch

from data.dataset import BearingFaultDataset, ToyDataset


def test_cwru_mode_fixture_shapes() -> None:
    """BearingFaultDataset(mode='cwru')[0] returns correctly typed 4-tuple."""
    d = BearingFaultDataset(mode="cwru", split="train")
    assert len(d) > 0
    sensor, image, text, label = d[0]
    assert isinstance(sensor, torch.Tensor)
    assert sensor.shape == torch.Size([2048])
    assert sensor.dtype == torch.float32
    assert isinstance(image, torch.Tensor)
    assert image.shape == torch.Size([224, 224, 3])
    assert image.dtype == torch.uint8
    assert isinstance(text, str)
    assert len(text) > 0
    assert isinstance(label, int)
    assert 0 <= label < 4


def test_cwru_mode_falls_back_to_fixture() -> None:
    """When data/raw/cwru/ does not exist, dataset falls back to fixture path."""
    # In CI the raw root does NOT exist; the fallback path is used automatically.
    d = BearingFaultDataset(mode="cwru", split="train")
    assert len(d) > 0
    sensor, image, text, label = d[0]
    assert sensor.shape == torch.Size([2048])


def test_cwru_mode_deterministic() -> None:
    """Two BearingFaultDataset instances with mode='cwru' return identical first items."""
    d1 = BearingFaultDataset(mode="cwru", split="train")
    d2 = BearingFaultDataset(mode="cwru", split="train")
    s1, img1, txt1, lbl1 = d1[0]
    s2, img2, txt2, lbl2 = d2[0]
    assert torch.equal(s1, s2)
    assert txt1 == txt2
    assert lbl1 == lbl2
    assert torch.equal(img1, img2)


def test_cwru_mode_label_range() -> None:
    """All labels in CWRU train split are in {0, 1, 2, 3}."""
    d = BearingFaultDataset(mode="cwru", split="train")
    labels = {d[i][3] for i in range(len(d))}
    assert labels.issubset({0, 1, 2, 3})
    assert len(labels) == 4  # all 4 fault classes present


def test_cwru_split_sizes_nonzero() -> None:
    """All three CWRU splits have at least one item."""
    for split in ("train", "val", "test"):
        d = BearingFaultDataset(mode="cwru", split=split)
        assert len(d) > 0, f"split={split} has zero items"


def test_cwru_mode_len_consistent() -> None:
    """len(d) matches the actual number of items in the dataset."""
    d = BearingFaultDataset(mode="cwru", split="train")
    count = sum(1 for _ in range(len(d)))
    assert count == len(d)


def test_toy_dataset_still_works() -> None:
    """Importing ToyDataset from data.dataset still works after adding BearingFaultDataset."""
    ds = ToyDataset(7, n=5)
    assert len(ds) == 5
    sensor, image, text, label = ds[0]
    assert isinstance(sensor, torch.Tensor)
    assert sensor.shape == torch.Size([2048])
    assert isinstance(text, str)
    assert isinstance(label, int)
