"""Run the full CWRU modality ablation with resume support."""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
_PROJECT_ROOT_STR = str(_PROJECT_ROOT)
if _PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT_STR)

from eval.ablation import run_single  # noqa: E402


_CONDITIONS = ("sensors-only", "vision+text", "all-three")
_CSV_HEADER = ["condition", "seed", "final_val_f1", "wall_time_s", "peak_memory_bytes"]


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the full CWRU ablation.")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--out-csv", default="eval/results_cwru.csv")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    out_csv = pathlib.Path(args.out_csv)
    existing = _read_existing_pairs(out_csv)

    if args.dry_run:
        for condition in _CONDITIONS:
            for seed in range(args.seeds):
                if (condition, seed) in existing:
                    print(f"SKIP condition={condition} seed={seed} (already in CSV)")
                else:
                    print(f"PLAN: condition={condition} seed={seed}")
        return 0

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = out_csv.exists()
    mode = "a" if file_exists else "w"
    with out_csv.open(mode, encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        if not file_exists:
            writer.writerow(_CSV_HEADER)
            csv_file.flush()

        for condition in _CONDITIONS:
            for seed in range(args.seeds):
                if (condition, seed) in existing:
                    print(f"SKIP condition={condition} seed={seed} (already in CSV)")
                    continue

                try:
                    result = run_single(
                        condition=condition,
                        seed=seed,
                        mode="cwru",
                        n_epochs=args.epochs,
                    )
                except Exception as exc:
                    print(f"ERROR condition={condition} seed={seed}: {exc}", file=sys.stderr)
                    return 1

                writer.writerow(
                    [
                        result["condition"],
                        result["seed"],
                        result["final_val_f1"],
                        result["wall_time_s"],
                        _csv_peak_memory(result["peak_memory_bytes"]),
                    ],
                )
                csv_file.flush()
                existing.add((condition, seed))
                print(_format_done(condition, seed, result))

    return 0


def _read_existing_pairs(csv_path: pathlib.Path) -> set[tuple[str, int]]:
    if not csv_path.exists():
        return set()

    pairs: set[tuple[str, int]] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            pairs.add((row["condition"], int(row["seed"])))
    return pairs


def _csv_peak_memory(value: object) -> object:
    if value is None:
        return ""
    return value


def _format_done(condition: str, seed: int, result: dict) -> str:
    peak_memory = result["peak_memory_bytes"]
    if peak_memory is None:
        peak_memory_text = "N/A"
    else:
        peak_memory_text = f"{peak_memory}B"
    return (
        f"DONE condition={condition} seed={seed} "
        f"f1={result['final_val_f1']} wall={result['wall_time_s']}s "
        f"peak_mem={peak_memory_text}"
    )


if __name__ == "__main__":
    raise SystemExit(_main())
