from __future__ import annotations

import json

from .llm import PlatformLLM


def score_length(text: str, length_limit: str) -> float:
    n = len("".join(text.split()))
    if length_limit == "short":
        return 1.0 if n <= 50 else 0.0
    if length_limit == "medium":
        return 1.0 if 51 <= n <= 140 else 0.0
    if length_limit == "long":
        return 1.0 if n >= 141 else 0.0
    raise ValueError(f"未知长度档位: {length_limit}")


def parse_scores(text: str) -> dict[str, float]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"模型未返回 JSON: {text}")

    data = json.loads(text[start:end + 1])
    out = {}
    for k in ("issue", "method", "stance"):
        v = float(data[k])
        if v not in {0.0, 0.5, 1.0}:
            raise ValueError(f"{k} 分数非法: {v}")
        out[k] = v
    return out


def build_geval_prompt(guidance: dict, generated_text: str) -> str:
    return f"""
你是一个文本语义一致性评估器。

请评估生成文本是否符合给定要求，只评估以下三个维度：

1. issue：生成文本是否围绕指定议题展开。
1 = 一致；0.5 = 基本相关但有偏题/遗漏/偷换；0 = 主要讨论其他议题

2. method：生成文本是否主要体现指定宣传手段。
1 = 指定宣传手段是主要表达策略；0.5 = 有体现但不明显；0 = 基本没有体现

3. stance：生成文本对指定目标的实际态度是否符合要求。
1 = 立场一致；0.5 = 立场不明确/混合/部分一致；0 = 立场明显相反

议题：
{guidance["issue"]}

指定宣传手段：
{guidance["propaganda_method"]}

指定立场：
{guidance["stance"]["label"]}

立场判断目标：
{guidance["stance"]["target"]}

生成文本：
{generated_text}

只能输出 JSON，不要解释，不要 Markdown：

{{
  "issue": 0,
  "method": 0,
  "stance": 0
}}
""".strip()


def build_alignscore_prompt(guidance: dict, generated_text: str) -> str:
    reference = {
        "issue": guidance["issue"],
        "propaganda_method": guidance["propaganda_method"],
        "stance": guidance["stance"]["label"],
        "stance_target": guidance["stance"]["target"],
    }

    return f"""
你是 AlignScore 风格的语义对齐评分器。

请把“引导条件”视为 reference，把“生成文本”视为 hypothesis，判断 hypothesis 是否与 reference 对齐。

分别给出三个 0/0.5/1 分数：
issue：是否讨论同一议题。
method：主要修辞策略是否与指定宣传手段对齐。
stance：对 stance_target 的态度是否与 stance 对齐。

引导条件：
{json.dumps(reference, ensure_ascii=False, indent=2)}

生成文本：
{generated_text}

只能输出 JSON：

{{
  "issue": 0,
  "method": 0,
  "stance": 0
}}
""".strip()


def evaluate_one(guidance: dict, generated_text: str, method: str = "geval") -> dict:
    llm = PlatformLLM()

    if method == "geval":
        prompt = build_geval_prompt(guidance, generated_text)
    elif method == "alignscore":
        prompt = build_alignscore_prompt(guidance, generated_text)
    else:
        raise ValueError("method 只能是 geval 或 alignscore")

    result = llm.chat(
        system="你是语义一致性评价器。只输出 JSON，不要输出解释、思考过程或 Markdown。",
        user=prompt,
        temperature=0.0,
        max_tokens=512,
    )

    scores = parse_scores(result.text)
    scores["length"] = score_length(generated_text, guidance["length_limit"])
    scores["overall"] = (
        scores["issue"] + scores["method"] + scores["stance"] + scores["length"]
    ) / 4

    return {
        "scores": scores,
        "usage": result.usage,
        "raw_eval": result.text,
    }
