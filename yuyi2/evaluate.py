from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ["MODEL_DIR"])

from model import load_model, predict
from yuyi2.eval_core import evaluate_one


MODEL_DIR = Path(os.environ["MODEL_DIR"])
TEST_DIR = Path(os.environ["TEST_DIR"])
RESULT_DIR = Path(os.environ["RESULT_DIR"])


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    files = sorted(glob.glob(str(TEST_DIR / "*.jsonl")))
    if not files:
        raise FileNotFoundError(f"测试目录没有 jsonl: {TEST_DIR}")

    cases = load_jsonl(Path(files[0]))
    model = load_model(str(MODEL_DIR))
    preds = predict(model, cases)

    method = os.environ.get("YUYI_EVAL_METHOD", "geval").lower().strip()

    details = []
    for case, pred in zip(cases, preds):
        generated_text = pred["generated_text"]
        result = evaluate_one(
            guidance=case["guidance"],
            generated_text=generated_text,
            method=method,
        )

        details.append({
            "sample_id": case["sample_id"],
            "generated_text": generated_text,
            **result,
        })

    acc = sum(x["scores"]["overall"] for x in details) / len(details) if details else 0.0

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    with (RESULT_DIR / "details.jsonl").open("w", encoding="utf-8") as f:
        for item in details:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with (RESULT_DIR / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump([{"name": "acc", "value": acc, "unit": ""}], f, ensure_ascii=False)


if __name__ == "__main__":
    main()
