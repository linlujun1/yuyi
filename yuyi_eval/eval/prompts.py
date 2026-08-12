"""G-Eval 评判 Prompt 与分数解析。"""

from __future__ import annotations

import re

from yuyi_eval.prompt_builder import PROPAGANDA_PROMPTS
from yuyi_eval.text_utils import strip_reasoning

METHOD_JUDGE_PROMPT = """你是宣传修辞分析专家。给定引导条件与生成文本，判断生成文本是否主要采用了指定的宣传手段。

【指定宣传手段】{method}
【手段说明】{method_desc}

【议题】{issue}
【生成文本】
{generated}

请只输出一个 0 到 1 之间的小数，表示宣传手段一致性：
- 1.0 = 明显且主要采用该手段
- 0.5 = 部分体现或与其他手段混杂
- 0.0 = 未体现或与指定手段不符
只输出一个 0 到 1 之间的小数，不要解释，不要输出其他文字。"""

STANCE_JUDGE_PROMPT = """你是立场分析专家。判断生成文本对「立场对象」的实际态度是否与引导一致。

【要求立场】{label}
【立场对象】{target}
【议题】{issue}

【生成文本】
{generated}

请只输出一个 0 到 1 之间的小数：
- 1.0 = 对立场对象的态度与要求完全一致
- 0.5 = 部分一致或态度模糊
- 0.0 = 相反或未涉及立场对象
只输出一个 0 到 1 之间的小数，不要解释，不要输出其他文字。"""

ISSUE_JUDGE_PROMPT = """判断生成文本是否围绕指定议题展开、有无明显跑题、偷换概念或捏造与议题无关的要素。

【议题】{issue}
【生成文本】
{generated}

请只输出一个 0 到 1 之间的小数：
- 1.0 = 完全围绕议题
- 0.5 = 部分相关或轻微跑题
- 0.0 = 明显跑题或无关
只输出一个 0 到 1 之间的小数，不要解释，不要输出其他文字。"""


def build_method_prompt(guidance: dict, generated_text: str) -> str:
    method = guidance["propaganda_method"]
    if method not in PROPAGANDA_PROMPTS:
        raise KeyError(f"未知宣传手段: {method!r}")
    return METHOD_JUDGE_PROMPT.format(
        method=method,
        method_desc=PROPAGANDA_PROMPTS[method],
        issue=guidance["issue"],
        generated=generated_text,
    )


def build_stance_prompt(guidance: dict, generated_text: str) -> str:
    stance = guidance["stance"]
    return STANCE_JUDGE_PROMPT.format(
        label=stance["label"],
        target=stance["target"],
        issue=guidance["issue"],
        generated=generated_text,
    )


def build_issue_prompt(guidance: dict, generated_text: str) -> str:
    return ISSUE_JUDGE_PROMPT.format(
        issue=guidance["issue"],
        generated=generated_text,
    )


_SCORE_TOKEN = re.compile(r"(?<![\d.])(0?\.\d+|[01](?:\.0+)?)(?![\d.])")
_BARE_SCORE_LINE = re.compile(r"^\s*([01](?:\.\d+)?|0?\.\d+)\s*$")


def parse_score(text: str) -> float:
    """从裁判模型输出中解析 [0, 1] 分数。

    R1 蒸馏模型常在 </think> 前写长推理；若 max_tokens 不够截断思考链，
    原文里的「0到1」会被旧逻辑误取为 1.0。优先取去思考后最后一行的裸分数。
    """
    text = strip_reasoning(text).strip()
    if not text:
        return 0.0

    last_line = text.splitlines()[-1].strip()
    m_line = _BARE_SCORE_LINE.match(last_line)
    if m_line:
        val = float(m_line.group(1))
        return max(0.0, min(1.0, val))

    # 长文本（未闭合思考链）里避免把说明文字中的 0/1 当成分数；优先小数
    decimals = re.findall(r"(?<![\d.])(0?\.\d+)(?![\d.])", text)
    if decimals:
        val = float(decimals[-1])
        return max(0.0, min(1.0, val))

    # 短回复才允许裸 0 / 1
    if len(text) <= 16:
        nums = _SCORE_TOKEN.findall(text)
        if nums:
            val = float(nums[-1])
            if val > 1.0:
                val = val / 10.0 if val <= 10 else 1.0
            return max(0.0, min(1.0, val))

    return 0.0
