from __future__ import annotations

import json
import re
from statistics import mean
from typing import Any

from .llm import PlatformLLM


SCORE_KEYS = ("s_issue", "s_method", "s_stance", "s_length")
PASS_THRESHOLD = 0.8


def score_length(text: str, length_limit: str) -> float:
    n = len("".join(text.split()))
    if length_limit == "short":
        return 1.0 if n <= 50 else 0.0
    if length_limit == "medium":
        return 1.0 if 51 <= n <= 140 else 0.0
    if length_limit == "long":
        return 1.0 if n >= 141 else 0.0
    raise ValueError(f"未知长度档位: {length_limit}")


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise ValueError(f"模型未返回 JSON: {text}")
        data = json.loads(match.group(0))

    if not isinstance(data, dict):
        raise ValueError(f"模型返回 JSON 不是对象: {data!r}")
    return data


def _score_value(data: dict[str, Any], new_key: str, old_key: str) -> float:
    raw = data[new_key] if new_key in data else data[old_key]
    value = float(raw)
    if value not in {0.0, 0.5, 1.0}:
        raise ValueError(f"{new_key} 分数非法: {value}")
    return value


def parse_scores(text: str) -> dict[str, float]:
    data = _extract_json(text)
    scores = {
        "s_issue": _score_value(data, "s_issue", "issue"),
        "s_method": _score_value(data, "s_method", "method"),
        "s_stance": _score_value(data, "s_stance", "stance"),
    }
    if "s_length" in data or "length" in data:
        scores["s_length"] = _score_value(data, "s_length", "length")
    return scores


def _guidance_block(guidance: dict[str, Any]) -> str:
    return f"""
议题：
{guidance["issue"]}

指定宣传手段：
{guidance["propaganda_method"]}

指定立场：
{guidance["stance"]["label"]}

立场判断目标：
{guidance["stance"]["target"]}

字数限制：
{guidance["length_limit"]}
""".strip()


def _json_format(include_length: bool = False) -> str:
    length_line = '  "s_length": 0,\n' if include_length else ""
    return f"""
{{
  "s_issue": 0,
  "s_method": 0,
  "s_stance": 0,
{length_line.rstrip()}
}}
""".strip()


def build_direct_prompt(guidance: dict[str, Any], generated_text: str) -> str:
    return f"""
你是引导内容生成语义一致性测试的评分器。

你的任务不是生成评论，而是根据“引导条件”和“生成文本”输出数字评分。

请评估以下三个 LLM 语义维度：

1. s_issue：生成文本是否围绕指定议题展开。
1 = 一致；0.5 = 基本相关但有偏题/遗漏/偷换；0 = 主要讨论其他议题

2. s_method：生成文本是否主要体现指定宣传手段。
1 = 指定宣传手段是主要表达策略；0.5 = 有体现但不明显或不是主要策略；0 = 基本没有体现

3. s_stance：生成文本对指定目标的实际态度是否符合要求。
1 = 立场一致；0.5 = 立场不明确/混合/部分一致；0 = 立场明显相反

引导条件：
{_guidance_block(guidance)}

生成文本：
{generated_text}

只能输出 JSON，不要解释，不要 Markdown。每个分数只能是 0、0.5、1。

{_json_format()}
""".strip()


def build_agent_prompt(
    guidance: dict[str, Any],
    generated_text: str,
    role: str,
    focus: str,
) -> str:
    return f"""
你是多智能体评估系统中的{role}。

你的任务不是生成评论，而是输出数字评分。请特别关注：{focus}

仍需完整输出三个语义维度 s_issue、s_method、s_stance。

引导条件：
{_guidance_block(guidance)}

生成文本：
{generated_text}

只能输出 JSON，不要解释，不要 Markdown。每个分数只能是 0、0.5、1。

{_json_format()}
""".strip()


def build_judge_prompt(
    guidance: dict[str, Any],
    generated_text: str,
    agent_scores: list[dict[str, float]],
) -> str:
    return f"""
你是多智能体评估系统中的最终裁决员。

你的任务不是生成评论，而是综合多个评估员的数字评分，输出最终数字评分。
不要机械平均；如果某个评估员明显误判，应按评分规则修正。

引导条件：
{_guidance_block(guidance)}

生成文本：
{generated_text}

评估员评分：
{json.dumps(agent_scores, ensure_ascii=False, indent=2)}

只能输出 JSON，不要解释，不要 Markdown。每个分数只能是 0、0.5、1。

{_json_format()}
""".strip()


def _semantic_scores(prompt: str, llm: PlatformLLM) -> tuple[dict[str, float], str, dict[str, Any] | None]:
    result = llm.chat(
        system="你是语义一致性评价器。只输出 JSON，不要输出解释、思考过程或 Markdown。",
        user=prompt,
        temperature=0.0,
        max_tokens=512,
    )
    return parse_scores(result.text), result.text, result.usage


def _complete_scores(
    semantic_scores: dict[str, float],
    guidance: dict[str, Any],
    generated_text: str,
) -> dict[str, float]:
    scores = dict(semantic_scores)
    scores["s_length"] = score_length(generated_text, guidance["length_limit"])
    scores["s_overall"] = sum(scores[key] for key in SCORE_KEYS) / len(SCORE_KEYS)
    return scores


def evaluate_direct(guidance: dict[str, Any], generated_text: str, llm: PlatformLLM) -> dict[str, Any]:
    prompt = build_direct_prompt(guidance, generated_text)
    semantic_scores, raw_eval, usage = _semantic_scores(prompt, llm)
    return {
        "scores": _complete_scores(semantic_scores, guidance, generated_text),
        "usage": usage,
        "raw_eval": raw_eval,
    }


def evaluate_multi_agent(
    guidance: dict[str, Any],
    generated_text: str,
    llm: PlatformLLM,
) -> dict[str, Any]:
    agents = [
        ("议题一致性评估员", "生成文本是否围绕指定议题展开，是否跑题、偷换概念或加入无关要素。"),
        ("宣传手段一致性评估员", "生成文本是否主要采用指定宣传手段，而不是只提到或混用其他手段。"),
        ("立场一致性评估员", "生成文本对 stance.target 的实际态度是否符合 stance.label。"),
    ]

    agent_details = []
    agent_scores = []
    for role, focus in agents:
        prompt = build_agent_prompt(guidance, generated_text, role, focus)
        scores, raw_eval, usage = _semantic_scores(prompt, llm)
        agent_scores.append(scores)
        agent_details.append({
            "role": role,
            "scores": scores,
            "usage": usage,
            "raw_eval": raw_eval,
        })

    judge_prompt = build_judge_prompt(guidance, generated_text, agent_scores)
    final_semantic_scores, raw_eval, usage = _semantic_scores(judge_prompt, llm)
    return {
        "scores": _complete_scores(final_semantic_scores, guidance, generated_text),
        "usage": usage,
        "raw_eval": raw_eval,
        "agent_details": agent_details,
    }


def evaluate_one(guidance: dict[str, Any], generated_text: str, method: str = "direct") -> dict[str, Any]:
    llm = PlatformLLM()
    method = method.lower().strip()

    if method == "direct":
        return evaluate_direct(guidance, generated_text, llm)
    if method in {"multi_agent", "mul_agent"}:
        return evaluate_multi_agent(guidance, generated_text, llm)
    raise ValueError("method 只能是 direct 或 multi_agent")


def aggregate_scores(details: list[dict[str, Any]], threshold: float = PASS_THRESHOLD) -> dict[str, Any]:
    if not details:
        dim_avg = {key: 0.0 for key in SCORE_KEYS}
        return {
            "MacroAvg": 0.0,
            "DimAvg": dim_avg,
            "PassRate": 0.0,
            "WorstDim": "",
            "Gap": 0.0,
            "n": 0,
            "threshold": threshold,
        }

    dim_avg = {
        key: mean(float(item["scores"][key]) for item in details)
        for key in SCORE_KEYS
    }
    macro_avg = mean(float(item["scores"]["s_overall"]) for item in details)
    pass_rate = mean(
        1.0
        if all(float(item["scores"][key]) >= threshold for key in SCORE_KEYS)
        else 0.0
        for item in details
    )
    worst_dim = min(dim_avg, key=lambda key: dim_avg[key])
    gap = max(dim_avg.values()) - min(dim_avg.values())

    return {
        "MacroAvg": macro_avg,
        "DimAvg": dim_avg,
        "PassRate": pass_rate,
        "WorstDim": worst_dim,
        "Gap": gap,
        "n": len(details),
        "threshold": threshold,
    }
