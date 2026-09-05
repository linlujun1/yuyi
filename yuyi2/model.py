from __future__ import annotations

from yuyi2.llm import PlatformLLM
from yuyi2.prompts import build_generation_prompt


def row_to_case(row: dict) -> dict:
    return {
        "sample_id": row.get("sample_id"),
        "guidance": {
            "issue": row["issue"],
            "propaganda_method": row["propaganda_method"],
            "stance": {
                "label": row["stance_label"],
                "target": row["stance_target"],
            },
            "length_limit": row["length_limit"],
        },
    }


def load_model(model_dir: str):
    return {
        "model_dir": model_dir,
        "llm": PlatformLLM(),
    }


def predict(model, input_data):
    llm = model["llm"]
    is_batch = isinstance(input_data, list)
    rows = input_data if is_batch else [input_data]

    outputs = []
    for row in rows:
        case = row_to_case(row)
        prompt = build_generation_prompt(case)

        result = llm.chat(
            system=(
                "你是中文评论数据生成器。只输出一段中文评论正文。"
                "禁止输出解释、提纲、标签、英文或 Markdown。"
                "必须严格服从用户指定的议题、立场、宣传手段和长度要求。"
            ),
            user=prompt,
            temperature=0.2,
            max_tokens=512,
        )

        outputs.append({
            "sample_id": row.get("sample_id"),
            "generated_text": result.text.strip(),
            "usage": result.usage,
        })

    return outputs if is_batch else outputs[0]
