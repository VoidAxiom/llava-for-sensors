from __future__ import annotations

from collections.abc import Sequence
import re

import numpy as np
import pytest
import torch

from data.dataset import ToyDataset
from data.synthetic import (
    IMAGE_SHAPE,
    N_CLASSES,
    SENSOR_LENGTH,
    TEXT_TEMPLATES,
    SyntheticSample,
    _sample_image,
    generate,
)

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def test_generate_default_shapes_and_types() -> None:
    samples = generate(n=8, seed=0)

    assert len(samples) == 8
    for sample in samples:
        assert sample["sensor"].shape == (SENSOR_LENGTH,)
        assert sample["sensor"].dtype == np.float32
        assert sample["image"].shape == IMAGE_SHAPE
        assert sample["image"].dtype == np.uint8
        assert isinstance(sample["text"], str)
        assert sample["label"] in range(N_CLASSES)

    with pytest.raises(ValueError, match="n must be non-negative"):
        generate(n=-1, seed=0)


def test_generate_is_deterministic_for_seed() -> None:
    first = generate(n=12, seed=123)
    second = generate(n=12, seed=123)
    different = generate(n=12, seed=124)

    for left, right in zip(first, second, strict=True):
        assert np.array_equal(left["sensor"], right["sensor"])
        assert np.array_equal(left["image"], right["image"])
        assert left["text"] == right["text"]
        assert left["label"] == right["label"]

    assert any(
        not np.array_equal(left["sensor"], right["sensor"])
        or left["text"] != right["text"]
        or left["label"] != right["label"]
        for left, right in zip(first, different, strict=True)
    )


def test_class_balance() -> None:
    samples = generate(n=1000, seed=0)
    counts = np.bincount(_labels(samples), minlength=N_CLASSES)

    assert all(240 <= int(count) <= 260 for count in counts)


def test_modal_ambiguity_patterns() -> None:
    assert np.array_equal(_sample_image(0), _sample_image(2))
    assert np.array_equal(_sample_image(1), _sample_image(3))
    assert not np.array_equal(_sample_image(0), _sample_image(1))

    assert TEXT_TEMPLATES[0] == TEXT_TEMPLATES[1]
    assert TEXT_TEMPLATES[2] != TEXT_TEMPLATES[3]
    assert set(TEXT_TEMPLATES[0]).isdisjoint(TEXT_TEMPLATES[3])


def test_toy_dataset_wraps_samples() -> None:
    samples = generate(n=8, seed=0)
    dataset = ToyDataset(samples)

    sensor, image, text, label = dataset[0]
    assert len(dataset) == 8
    assert isinstance(sensor, torch.Tensor)
    assert isinstance(image, torch.Tensor)
    assert sensor.dtype == torch.float32
    assert image.dtype == torch.uint8
    assert sensor.shape == (SENSOR_LENGTH,)
    assert image.shape == IMAGE_SHAPE
    assert text == samples[0]["text"]
    assert label == int(samples[0]["label"])
    assert torch.equal(sensor, torch.tensor(samples[0]["sensor"], dtype=torch.float32))
    assert torch.equal(image, torch.tensor(samples[0]["image"], dtype=torch.uint8))


def test_toy_dataset_can_generate_from_seed() -> None:
    dataset = ToyDataset(7, n=12)
    expected = generate(n=12, seed=7)

    sensor, image, text, label = dataset[0]
    assert len(dataset) == 12
    assert torch.equal(sensor, torch.tensor(expected[0]["sensor"], dtype=torch.float32))
    assert torch.equal(image, torch.tensor(expected[0]["image"], dtype=torch.uint8))
    assert text == expected[0]["text"]
    assert label == int(expected[0]["label"])


def test_cross_modal_required_property() -> None:
    samples = generate(n=1000, seed=0)
    y = _labels(samples)
    train_idx, test_idx = _stratified_split(samples, test_size=200, seed=0)

    sensor_feats = _sensor_features(samples)
    image_feats = _image_features(samples)
    text_feats = _text_features(samples, train_idx, test_idx)

    f1_sensor = _fit_score(sensor_feats, y, train_idx, test_idx)
    f1_image = _fit_score(image_feats, y, train_idx, test_idx)
    f1_text = _fit_score(text_feats, y, train_idx, test_idx)

    assert max(f1_sensor, f1_image, f1_text) < 0.90


def _labels(samples: Sequence[SyntheticSample]) -> np.ndarray:
    return np.array([sample["label"] for sample in samples], dtype=np.int64)


def _stratified_split(
    samples: Sequence[SyntheticSample], test_size: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    y = _labels(samples)
    if HAS_SKLEARN:
        indices = np.arange(len(samples))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
            stratify=y,
        )
        return np.array(train_idx, dtype=np.int64), np.array(test_idx, dtype=np.int64)

    by_class = {class_id: np.flatnonzero(y == class_id) for class_id in range(N_CLASSES)}
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    test_idx: list[int] = []
    n_test_per_class = test_size // N_CLASSES
    for class_id in range(N_CLASSES):
        class_indices = by_class[class_id].copy()
        rng.shuffle(class_indices)
        test_idx.extend(int(index) for index in class_indices[:n_test_per_class])
        train_idx.extend(int(index) for index in class_indices[n_test_per_class:])
    return np.array(train_idx, dtype=np.int64), np.array(test_idx, dtype=np.int64)


def _sensor_features(samples: Sequence[SyntheticSample]) -> np.ndarray:
    rows = [
        np.abs(np.fft.rfft(sample["sensor"])[:32]).astype(np.float32)
        for sample in samples
    ]
    return np.vstack(rows)


def _image_features(samples: Sequence[SyntheticSample]) -> np.ndarray:
    rows = [
        sample["image"][::28, ::28, :].astype(np.float32).reshape(-1) / 255.0
        for sample in samples
    ]
    return np.vstack(rows)


def _text_features(
    samples: Sequence[SyntheticSample], train_idx: np.ndarray, test_idx: np.ndarray
) -> np.ndarray:
    texts = [sample["text"] for sample in samples]
    if HAS_SKLEARN:
        vectorizer = TfidfVectorizer(max_features=50)
        train_features = vectorizer.fit_transform([texts[int(index)] for index in train_idx])
        test_features = vectorizer.transform([texts[int(index)] for index in test_idx])
        features = np.zeros((len(samples), train_features.shape[1]), dtype=np.float32)
        features[train_idx] = train_features.toarray().astype(np.float32)
        features[test_idx] = test_features.toarray().astype(np.float32)
        return features
    return _manual_text_features(texts, train_idx, test_idx)


def _manual_text_features(
    texts: Sequence[str], train_idx: np.ndarray, test_idx: np.ndarray
) -> np.ndarray:
    token_re = re.compile(r"[a-z]+(?:-[a-z]+)?")
    train_tokens = [
        [token for token in token_re.findall(texts[int(index)].lower())]
        for index in train_idx
    ]
    vocab = sorted({token for tokens in train_tokens for token in tokens})
    features = np.zeros((len(texts), len(vocab)), dtype=np.float32)
    vocab_index = {token: index for index, token in enumerate(vocab)}

    for row_index in np.concatenate([train_idx, test_idx]):
        for token in token_re.findall(texts[int(row_index)].lower()):
            column = vocab_index.get(token)
            if column is not None:
                features[int(row_index), column] += 1.0
    return features


def _fit_score(
    features: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> float:
    x_train, x_test = _standardize_train_test(features[train_idx], features[test_idx])
    y_train = y[train_idx]
    y_test = y[test_idx]

    if HAS_SKLEARN:
        model = LogisticRegression(max_iter=1000)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        return float(f1_score(y_test, y_pred, average="macro"))

    weights, bias = _fit_lr(x_train, y_train, n_classes=N_CLASSES)
    y_pred = _predict_lr(x_test, weights, bias)
    return _macro_f1(y_test, y_pred, n_classes=N_CLASSES)


def _standardize_train_test(
    x_train: np.ndarray, x_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (x_train - mean) / std, (x_test - mean) / std


def _softmax(x: np.ndarray) -> np.ndarray:
    exp = np.exp(x - x.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def _fit_lr(
    x: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    lr: float = 0.1,
    n_iter: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    n_samples, n_features = x.shape
    weights = np.zeros((n_features, n_classes), dtype=np.float64)
    bias = np.zeros(n_classes, dtype=np.float64)

    for _ in range(n_iter):
        probs = _softmax(x @ weights + bias)
        probs[np.arange(n_samples), y] -= 1.0
        weights -= lr * (x.T @ probs) / n_samples
        bias -= lr * probs.mean(axis=0)

    return weights, bias


def _predict_lr(x: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return _softmax(x @ weights + bias).argmax(axis=1)


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    f1s: list[float] = []
    for class_id in range(n_classes):
        true_class = y_true == class_id
        pred_class = y_pred == class_id
        tp = int((true_class & pred_class).sum())
        fp = int((~true_class & pred_class).sum())
        fn = int((true_class & ~pred_class).sum())
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        if precision + recall == 0.0:
            f1s.append(0.0)
        else:
            f1s.append(2.0 * precision * recall / (precision + recall))
    return float(np.mean(f1s))
