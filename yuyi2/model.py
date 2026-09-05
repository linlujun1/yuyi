from __future__ import annotations

from typing import Any


def load_model(model_dir: str) -> dict[str, Any]:
    return {"model_dir": model_dir}


def _predict_one(row: dict[str, Any]) -> dict[str, Any]:
    generated_text = (
        row.get("generated_text")
        or row.get("prediction")
        or row.get("output")
        or row.get("text")
        or ""
    )
    return {
        "sample_id": row.get("sample_id") or row.get("id"),
        "generated_text": str(generated_text).strip(),
    }


def predict(model: dict[str, Any], input_data: Any) -> Any:
    if isinstance(input_data, list):
        return [_predict_one(row) for row in input_data]
    return _predict_one(input_data)
