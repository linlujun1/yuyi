from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path


DEFAULT_INPUT_GLOB = "data/evaluation/*.jsonl"
DEFAULT_OUTPUT_PATH = Path("data/evaluation_summary.csv")
SCORE_KEYS = [
    "issue",
    "method",
    "stance",
    "length",
    "overall",
]
DIMENSION_KEYS = [
    "issue",
    "method",
    "stance",
    "length",
]


def load_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def infer_record_value(
    records: list[dict],
    key: str,
    default: str = "unknown",
) -> str:
    for record in records:
        value = record.get(key)
        if value is not None:
            return str(value)

    return default


def summarize_file(path: Path, threshold: float) -> dict:
    records = load_jsonl(path)

    if not records:
        return {
            "input_file": str(path),
            "n": 0,
            "eval_method": "unknown",
            "source_model": "unknown",
            "evaluator_model": "unknown",
            "with_target": "unknown",
            **{key: 0.0 for key in SCORE_KEYS},
            "pass_rate": 0.0,
            "worst_dim": "unknown",
            "gap": 0.0,
        }

    scores_by_key = {
        key: [
            float(record.get("scores", {}).get(key, 0.0))
            for record in records
        ]
        for key in SCORE_KEYS
    }

    dim_avgs = {
        key: average(scores_by_key[key])
        for key in DIMENSION_KEYS
    }
    worst_dim = min(
        dim_avgs,
        key=lambda key: dim_avgs[key],
    )
    pass_count = 0

    for record in records:
        scores = record.get("scores", {})
        if all(
            float(scores.get(key, 0.0)) >= threshold
            for key in DIMENSION_KEYS
        ):
            pass_count += 1

    return {
        "input_file": str(path),
        "n": len(records),
        "eval_method": infer_record_value(records, "eval_method"),
        "source_model": infer_record_value(records, "source_model"),
        "evaluator_model": infer_record_value(records, "evaluator_model"),
        "with_target": infer_record_value(records, "with_target"),
        **{
            key: round(average(scores_by_key[key]), 6)
            for key in SCORE_KEYS
        },
        "pass_rate": round(pass_count / len(records), 6),
        "worst_dim": worst_dim,
        "gap": round(max(dim_avgs.values()) - min(dim_avgs.values()), 6),
    }


def resolve_inputs(patterns: list[str] | None) -> list[Path]:
    if not patterns:
        patterns = [DEFAULT_INPUT_GLOB]

    paths: list[Path] = []

    for pattern in patterns:
        path = Path(pattern)
        if path.is_file():
            paths.append(path)
            continue

        matched = sorted(Path(match) for match in glob.glob(pattern))
        if matched:
            paths.extend(path for path in matched if path.is_file())
            continue

    seen = set()
    unique_paths = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_paths.append(path)

    return unique_paths


def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "input_file",
        "n",
        "eval_method",
        "source_model",
        "evaluator_model",
        "with_target",
        "issue",
        "method",
        "stance",
        "length",
        "overall",
        "pass_rate",
        "worst_dim",
        "gap",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict]) -> None:
    for row in rows:
        print(
            f"{row['eval_method']} | "
            f"{row['source_model']} -> {row['evaluator_model']} | "
            f"n={row['n']} | "
            f"overall={row['overall']:.4f} | "
            f"issue={row['issue']:.4f} | "
            f"method={row['method']:.4f} | "
            f"stance={row['stance']:.4f} | "
            f"length={row['length']:.4f} | "
            f"pass_rate={row['pass_rate']:.4f} | "
            f"worst={row['worst_dim']}"
        )


def summarize_results(
    inputs: list[str] | None = None,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    threshold: float = 0.8,
) -> Path:
    paths = resolve_inputs(inputs)

    if not paths:
        raise FileNotFoundError(
            "没有找到评价结果文件；"
            f"默认搜索: {DEFAULT_INPUT_GLOB}"
        )

    rows = [
        summarize_file(path, threshold)
        for path in paths
    ]

    write_csv(rows, output_path)
    print_summary(rows)
    print(f"汇总文件: {output_path.resolve()}")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="*",
        help=(
            "评价 JSONL 文件或 glob；不填则默认读取 "
            f"{DEFAULT_INPUT_GLOB}"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
    )

    args = parser.parse_args()

    summarize_results(
        inputs=args.inputs,
        output_path=args.output,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
