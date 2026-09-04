from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path


FIELDNAMES = [
    "sample_id",
    "topic_id",
    "topic_type",
    "source_backend",
    "source_model",
    "issue",
    "propaganda_method",
    "stance_label",
    "stance_target",
    "length_limit",
    "generated_text",
    "char_count",
]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def flatten(row: dict) -> dict:
    guidance = row["guidance"]
    stance = guidance["stance"]
    generated_text = str(row.get("generated_text", "")).strip()
    return {
        "sample_id": row.get("sample_id", ""),
        "topic_id": row.get("topic_id", ""),
        "topic_type": row.get("topic_type", ""),
        "source_backend": row.get("backend", ""),
        "source_model": row.get("model", ""),
        "issue": guidance.get("issue", ""),
        "propaganda_method": guidance.get("propaganda_method", ""),
        "stance_label": stance.get("label", ""),
        "stance_target": stance.get("target", ""),
        "length_limit": guidance.get("length_limit", ""),
        "generated_text": generated_text,
        "char_count": row.get("char_count") or len(generated_text),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", default="/user_home/linlujun/linlujun/yuyi2/data")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()

    rows = [flatten(row) for row in load_jsonl(Path(args.input))]
    random.Random(args.seed).shuffle(rows)

    n_test = max(1, int(len(rows) * args.test_ratio))
    test_rows = rows[:n_test]
    train_rows = rows[n_test:]

    out = Path(args.out)
    write_csv(out / "train" / "train.csv", train_rows)
    write_csv(out / "test" / "test.csv", test_rows)

    zip_path = out.parent / "yuyi2_dataset.zip"
    if zip_path.exists():
        zip_path.unlink()
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", root_dir=out)

    print("total:", len(rows))
    print("train:", len(train_rows), out / "train" / "train.csv")
    print("test:", len(test_rows), out / "test" / "test.csv")
    print("zip:", zip_path)


if __name__ == "__main__":
    main()
