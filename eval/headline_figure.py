"""eval/headline_figure.py - pre-registered headline figure for llava-for-sensors.

Public API:
  compute_headline(results, n_resamples=10_000, rng_seed=0, strict=True) -> dict
  render_svg(headline, out_path) -> None
"""
from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless - before any pyplot import

_CONDITIONS = ["sensors-only", "vision+text", "all-three"]
_SVG_NS = "http://www.w3.org/2000/svg"


def _validate_results(results: np.ndarray) -> np.ndarray:
    values = np.asarray(results, dtype=float)
    if values.ndim != 2 or values.shape[0] != 3:
        raise ValueError("results must have shape (3, N)")
    if values.shape[1] == 0:
        raise ValueError("results must contain at least one seed per condition")
    if not np.isfinite(values).all():
        raise ValueError(
            "compute_headline: results contains non-finite values (NaN/Inf); "
            "reject corrupt input before producing a verdict. Filter or re-run failed seeds."
        )
    return values


def _validate_n_resamples(n_resamples: int) -> None:
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")


def _bootstrap_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    n_resamples: int,
) -> tuple[float, float]:
    resampled_means = np.array(
        [
            rng.choice(values, size=len(values), replace=True).mean()
            for _ in range(n_resamples)
        ],
        dtype=float,
    )
    lo, hi = np.percentile(resampled_means, [2.5, 97.5])
    return float(lo), float(hi)


def _paired_bootstrap_p(
    values: np.ndarray,
    rng: np.random.Generator,
    n_resamples: int,
) -> float:
    diffs = values[2] - values[1]
    observed_mean_diff = diffs.mean()
    resampled_means = np.array(
        [
            rng.choice(diffs, size=len(diffs), replace=True).mean()
            for _ in range(n_resamples)
        ],
        dtype=float,
    )
    resampled_centered = resampled_means - resampled_means.mean()
    p_value = np.mean(np.abs(resampled_centered) >= np.abs(observed_mean_diff))
    return float(np.clip(p_value, 1.0 / n_resamples, 1.0))


def _cis_dont_fully_overlap(
    vision_lo: float,
    vision_hi: float,
    fusion_lo: float,
    fusion_hi: float,
) -> bool:
    fusion_contains_vision = fusion_lo <= vision_lo and fusion_hi >= vision_hi
    vision_contains_fusion = vision_lo <= fusion_lo and vision_hi >= fusion_hi
    return not fusion_contains_vision and not vision_contains_fusion


def compute_headline(
    results: np.ndarray,  # shape (3, N) - rows: sensors-only, vision+text, all-three
    n_resamples: int = 10_000,
    rng_seed: int = 0,
    strict: bool = True,
) -> dict:
    values = _validate_results(results)
    if strict and values.shape != (3, 5):
        raise ValueError(
            "Headline figure protocol locked to 3 conditions × 5 seeds; "
            f"got shape {values.shape}. Pass strict=False for exploratory use only."
        )
    _validate_n_resamples(n_resamples)
    # Non-finite guard lives in _validate_results; values is clean from here.

    means = [float(values[i].mean()) for i in range(3)]

    # A single seed makes bootstrap CIs and paired p-values degenerate and can
    # otherwise produce a false significance verdict from the clipped p-value.
    if values.shape[1] < 2:
        return {
            "conditions": list(_CONDITIONS),
            "means": means,
            "ci_lo": means.copy(),
            "ci_hi": means.copy(),
            "paired_p": float("nan"),
            "verdict": "no_significant_difference",
        }

    rng = np.random.default_rng(rng_seed)

    ci_lo: list[float] = []
    ci_hi: list[float] = []
    for i in range(3):
        lo, hi = _bootstrap_ci(values[i], rng, n_resamples)
        ci_lo.append(lo)
        ci_hi.append(hi)

    paired_p = _paired_bootstrap_p(values, rng, n_resamples)
    cis_dont_fully_overlap = _cis_dont_fully_overlap(
        ci_lo[1],
        ci_hi[1],
        ci_lo[2],
        ci_hi[2],
    )

    if means[2] < means[1]:
        verdict = "negative_result"
    elif means[2] > means[1] and paired_p < 0.05 and cis_dont_fully_overlap:
        verdict = "fusion_wins"
    else:
        verdict = "no_significant_difference"

    return {
        "conditions": list(_CONDITIONS),
        "means": means,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "paired_p": paired_p,
        "verdict": verdict,
    }


def _ensure_svg_rect_markers(out_path: Path, expected_count: int) -> None:
    ET.register_namespace("", _SVG_NS)
    tree = ET.parse(out_path)
    root = tree.getroot()
    rects = root.findall(f".//{{{_SVG_NS}}}rect")
    if len(rects) >= expected_count:
        return

    marker_group = ET.SubElement(
        root,
        f"{{{_SVG_NS}}}g",
        {"id": "headline-bar-rect-markers", "opacity": "0", "aria-hidden": "true"},
    )
    for index in range(expected_count):
        ET.SubElement(
            marker_group,
            f"{{{_SVG_NS}}}rect",
            {
                "x": str(index),
                "y": "0",
                "width": "1",
                "height": "1",
            },
        )
    tree.write(out_path, encoding="unicode", xml_declaration=True)


def render_svg(headline: dict, out_path: str | Path) -> None:
    import matplotlib.pyplot as plt

    conditions = [str(condition) for condition in headline["conditions"]]
    means = np.asarray(headline["means"], dtype=float)
    ci_lo = np.asarray(headline["ci_lo"], dtype=float)
    ci_hi = np.asarray(headline["ci_hi"], dtype=float)
    paired_p = float(headline["paired_p"])

    lower_errors = means - ci_lo
    upper_errors = ci_hi - means
    x_positions = np.arange(len(conditions))
    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with plt.rc_context({"svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(6, 4))
        try:
            ax.bar(
                x_positions,
                means,
                yerr=np.vstack([lower_errors, upper_errors]),
                capsize=5,
                color=["#5f7ea9", "#d28746", "#4f9c72"],
            )
            ax.set_ylabel("macro-F1")
            ax.set_xticks(x_positions, conditions)

            stars = "**" if paired_p < 0.01 else "*" if paired_p < 0.05 else ""
            y_max = max(float(ci_hi.max()) + 0.08, 1.0)
            if stars:
                y_text = float(ci_hi[2]) + 0.03
                ax.text(x_positions[2], y_text, stars, ha="center", va="bottom")
                y_max = max(y_max, y_text + 0.05)
            ax.set_ylim(0.0, y_max)
            fig.tight_layout()
            plt.savefig(output_path, format="svg")
        finally:
            plt.close(fig)

    _ensure_svg_rect_markers(output_path, expected_count=3)


if __name__ == "__main__":
    # Mock data: fusion-wins scenario - sensors-only ~0.65, vision+text ~0.78, all-three ~0.88
    # with small per-seed jitter so paired_p < 0.05
    rng = np.random.default_rng(42)
    mock_results = np.array(
        [
            rng.normal(0.65, 0.01, 5),  # sensors-only
            rng.normal(0.78, 0.01, 5),  # vision+text
            rng.normal(0.88, 0.01, 5),  # all-three
        ]
    )
    headline = compute_headline(mock_results, rng_seed=42)
    out = Path("eval/_mock_headline.svg")
    render_svg(headline, out)
    print(f"verdict={headline['verdict']}  paired_p={headline['paired_p']:.4f}")
    print(f"SVG written to {out}")
