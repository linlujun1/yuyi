"""宣传手段 / 立场：G-Eval 打分（经 JudgeClient，便于切换模型）。"""

from __future__ import annotations

from yuyi_eval.eval.judge import JudgeClient
from yuyi_eval.eval.prompts import build_method_prompt, build_stance_prompt, parse_score


class GEvalDimensions:
    def __init__(self, judge: JudgeClient, *, max_tokens: int = 1024):
        self.judge = judge
        # R1 系列需较大 max_tokens 以容纳思考链；最终分由 parse_score 截取
        self.max_tokens = max_tokens

    def score_method(self, guidance: dict, generated_text: str) -> float:
        prompt = build_method_prompt(guidance, generated_text)
        return parse_score(
            self.judge.complete(prompt, temperature=0.0, max_tokens=self.max_tokens)
        )

    def score_stance(self, guidance: dict, generated_text: str) -> float:
        prompt = build_stance_prompt(guidance, generated_text)
        return parse_score(
            self.judge.complete(prompt, temperature=0.0, max_tokens=self.max_tokens)
        )
