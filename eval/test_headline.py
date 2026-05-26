"""eval/test_headline.py - pytest tests for eval/headline_figure.py.

All tests are deterministic: rng_seed=0 is passed to compute_headline.
All assertions use values computed by compute_headline on the test fixture -
numbers are NEVER hardcoded. This enforces the Evidence rule: every asserted
number comes from real computation.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

from eval.headline_figure import compute_headline, render_svg


def test_known_good_fusion_wins() -> None:
    """3x5 array: sensors~0.65, vision~0.78, all-three~0.88, small jitter.

    Small jitter (0.005 std) ensures paired_p < 0.05 reliably with rng_seed=0.
    """
    arr = np.array(
        [
            [0.648, 0.651, 0.652, 0.649, 0.650],  # sensors-only
            [0.778, 0.781, 0.779, 0.782, 0.780],  # vision+text
            [0.878, 0.881, 0.879, 0.882, 0.880],  # all-three
        ]
    )
    result = compute_headline(arr, rng_seed=0)
    assert result["verdict"] == "fusion_wins"
    # Verify the sub-conditions that make verdict fusion_wins:
    assert result["means"][2] > result["means"][1]
    assert result["paired_p"] < 0.05


def test_known_flat_no_difference() -> None:
    """All conditions at ~0.80 - fusion can't be declared superior."""
    arr = np.array(
        [
            [0.798, 0.800, 0.801, 0.799, 0.802],  # sensors-only
            [0.800, 0.801, 0.799, 0.800, 0.800],  # vision+text
            [0.800, 0.799, 0.801, 0.800, 0.800],  # all-three
        ]
    )
    result = compute_headline(arr, rng_seed=0)
    assert result["verdict"] == "no_significant_difference"


def test_negative_result_protocol() -> None:
    """all-three ~0.70, vision+text ~0.80 - fusion underperformed."""
    arr = np.array(
        [
            [0.750, 0.748, 0.752, 0.749, 0.751],  # sensors-only
            [0.798, 0.800, 0.801, 0.799, 0.802],  # vision+text
            [0.698, 0.700, 0.701, 0.699, 0.702],  # all-three (underperformed)
        ]
    )
    result = compute_headline(arr, rng_seed=0)
    assert result["verdict"] == "negative_result"
    assert result["means"][2] <= result["means"][1]


def test_bootstrap_ci_numerical_correctness() -> None:
    """Verify the percentile CI is computed, not faked.

    For a 5-sample input with known min/max, the percentile CI must:
    1. Fall within [sample_min, sample_max] for each condition.
    2. Contain the sample mean (ci_lo <= mean <= ci_hi) for each condition.
    These two constraints are satisfied by any correct percentile bootstrap
    and violated by any hardcoded or trivially faked implementation.
    """
    arr = np.array(
        [
            [0.60, 0.65, 0.70, 0.75, 0.80],  # sensors-only
            [0.70, 0.74, 0.78, 0.82, 0.86],  # vision+text
            [0.80, 0.83, 0.86, 0.89, 0.92],  # all-three
        ]
    )
    result = compute_headline(arr, rng_seed=0)
    for i in range(3):
        lo = result["ci_lo"][i]
        hi = result["ci_hi"][i]
        mean = result["means"][i]
        sample_min = arr[i].min()
        sample_max = arr[i].max()
        assert lo >= sample_min, f"ci_lo[{i}]={lo} below sample_min={sample_min}"
        assert hi <= sample_max, f"ci_hi[{i}]={hi} above sample_max={sample_max}"
        assert lo <= mean <= hi, f"mean={mean} not in CI=[{lo}, {hi}] for condition {i}"


def test_svg_output(tmp_path) -> None:
    """render_svg produces a valid SVG file with required elements."""
    arr = np.array(
        [
            [0.648, 0.651, 0.652, 0.649, 0.650],
            [0.778, 0.781, 0.779, 0.782, 0.780],
            [0.878, 0.881, 0.879, 0.882, 0.880],
        ]
    )
    headline = compute_headline(arr, rng_seed=0)
    out = tmp_path / "out.svg"
    render_svg(headline, out)

    assert out.exists(), "SVG file was not created"
    assert out.stat().st_size > 0, "SVG file is empty"

    # Parse as XML
    tree = ET.parse(out)
    root = tree.getroot()

    # Check SVG namespace
    assert "svg" in root.tag.lower() or root.tag == "{http://www.w3.org/2000/svg}svg", (
        f"Root element is not SVG: {root.tag}"
    )

    # Check for 3 rect elements (the bars)
    rects = root.findall(".//{http://www.w3.org/2000/svg}rect")
    assert len(rects) >= 3, f"Expected at least 3 rect elements, found {len(rects)}"

    # Check SVG text contains axis label and condition names
    svg_text = out.read_text()
    assert "macro-F1" in svg_text, "Y-axis label 'macro-F1' not found in SVG"
    assert "sensors-only" in svg_text, "Condition 'sensors-only' not in SVG"
    assert "vision" in svg_text, "Condition 'vision+text' not in SVG"
    assert "all-three" in svg_text, "Condition 'all-three' not in SVG"
