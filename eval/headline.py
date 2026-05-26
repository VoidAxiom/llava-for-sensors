"""Headline acceptance gate for ablation CSV results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from eval.headline_figure import compute_headline, render_svg


_CONDITIONS = ["sensors-only", "vision+text", "all-three"]


class PhaseAcceptanceError(Exception):
    """Raised when ablation results fail the phase acceptance gate."""


def run_headline_from_csv(
    csv_path: str,
    out_svg: str = "docs/figures/headline.svg",
) -> dict:
    results = _read_results_csv(Path(csv_path))
    n_seeds = results.shape[1]
    if n_seeds != 5:
        raise ValueError(
            f"Headline gate requires exactly 5 seeds per condition; got {n_seeds}. "
            "Run the full ablation (n_seeds=5) before applying the gate."
        )
    headline = compute_headline(results, n_resamples=10000, rng_seed=0, strict=True)
    render_svg(headline, out_svg)

    all_three_mean = float(headline["means"][2])
    vision_text_mean = float(headline["means"][1])
    sensors_only_mean = float(headline["means"][0])
    gap_vt = all_three_mean - vision_text_mean
    gap_so = all_three_mean - sensors_only_mean

    failures: list[str] = []
    if not all_three_mean > vision_text_mean:
        failures.append(
            f"all-three mean {all_three_mean:.6f} is not greater than "
            f"vision+text mean {vision_text_mean:.6f}"
        )
    if not all_three_mean > sensors_only_mean:
        failures.append(
            f"all-three mean {all_three_mean:.6f} is not greater than "
            f"sensors-only mean {sensors_only_mean:.6f}"
        )
    if not gap_vt > 0.15:
        failures.append(f"all-three minus vision+text gap {gap_vt:.6f} is not > 0.15")
    if not gap_so > 0.15:
        failures.append(f"all-three minus sensors-only gap {gap_so:.6f} is not > 0.15")
    if failures:
        raise PhaseAcceptanceError("Headline acceptance failed: " + "; ".join(failures))

    return {**headline, "acceptance": "passed", "gap_vt": gap_vt, "gap_so": gap_so}


def _read_results_csv(csv_path: Path) -> np.ndarray:
    values_by_condition: dict[str, list[tuple[int, float]]] = {
        condition: [] for condition in _CONDITIONS
    }
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            condition = row["condition"]
            if condition not in values_by_condition:
                continue
            values_by_condition[condition].append(
                (int(row["seed"]), float(row["final_val_f1"]))
            )

    rows: list[list[float]] = []
    expected_len: int | None = None
    for condition in _CONDITIONS:
        condition_rows = sorted(values_by_condition[condition], key=lambda item: item[0])
        values = [value for _, value in condition_rows]
        if not values:
            raise ValueError(f"missing results for condition: {condition}")
        if expected_len is None:
            expected_len = len(values)
        elif len(values) != expected_len:
            raise ValueError("all conditions must have the same number of seeds")
        rows.append(values)

    return np.asarray(rows, dtype=float)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Render and gate the headline figure.")
    parser.add_argument("--csv", default="eval/results_toy.csv")
    parser.add_argument("--out", default="docs/figures/headline.svg")
    args = parser.parse_args()

    run_headline_from_csv(csv_path=args.csv, out_svg=args.out)


if __name__ == "__main__":
    _main()
