"""G-Eval 风格 LLM-as-judge（宣传手段 / 立场维度）."""

from __future__ import annotations

import re

from yuyi_eval.text_utils import strip_reasoning

from yuyi_eval.llm_router import LLMRouter
from yuyi_eval.prompt_builder import PROPAGANDA_PROMPTS

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


def _parse_score(text: str) -> float:
    text = strip_reasoning(text)
    nums = re.findall(r"(?<![\d.])(0?\.\d+|[01](?:\.0+)?)(?![\d.])", text)
    if not nums:
        m = re.search(r"(\d+(?:\.\d+)?)", text)
        if not m:
            return 0.0
        val = float(m.group(1))
    else:
        val = float(nums[-1])
    if val > 1.0:
        val = val / 10.0 if val <= 10 else 1.0
    return max(0.0, min(1.0, val))


class GEvalJudge:
    def __init__(self, router: LLMRouter, judge_tmodel: str = "gpt4o-mini"):
        self.router = router
        self.judge_tmodel = judge_tmodel

    def score_method(self, guidance: dict, generated_text: str) -> float:
        method = guidance["propaganda_method"]
        prompt = METHOD_JUDGE_PROMPT.format(
            method=method,
            method_desc=PROPAGANDA_PROMPTS[method],
            issue=guidance["issue"],
            generated=generated_text,
        )
        resp = self.router.chat(
            self.judge_tmodel,
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=16,
        )
        return _parse_score(resp)

    def score_stance(self, guidance: dict, generated_text: str) -> float:
        stance = guidance["stance"]
        prompt = STANCE_JUDGE_PROMPT.format(
            label=stance["label"],
            target=stance["target"],
            issue=guidance["issue"],
            generated=generated_text,
        )
        resp = self.router.chat(
            self.judge_tmodel,
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=16,
        )
        return _parse_score(resp)

    def evaluate_sample(
        self,
        guidance: dict,
        generated_text: str,
        *,
        use_alignscore_issue: bool = False,
        alignscore=None,
    ) -> dict[str, float]:
        scores = {
            "s_method": self.score_method(guidance, generated_text),
            "s_stance": self.score_stance(guidance, generated_text),
        }
        if use_alignscore_issue and alignscore is not None:
            scores["s_issue"] = alignscore.score_issue_consistency(
                guidance["issue"], generated_text
            )
        return scores
