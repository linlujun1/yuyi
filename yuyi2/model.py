from __future__ import annotations


def load_model(model_dir: str):
    return {"model_dir": model_dir}


def predict(model, input_data):
    # 当前阶段：数据集中已经包含 generated_text，所以 predict 直接返回待评估文本。
    # 下一步重新生成评论数据集后，再把这里改成真正调用 LLM 生成评论。
    if isinstance(input_data, list):
        return [
            {
                "sample_id": item["sample_id"],
                "generated_text": item["generated_text"],
            }
            for item in input_data
        ]

    return {
        "sample_id": input_data["sample_id"],
        "generated_text": input_data["generated_text"],
    }
