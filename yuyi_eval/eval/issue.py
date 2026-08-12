"""议题一致性：AlignScore 为主，LLM 兜底（LLM 调用暂未接入）。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Protocol

from yuyi_eval.eval.judge import JudgeClient
from yuyi_eval.eval.prompts import build_issue_prompt, parse_score

ROOT = Path(__file__).resolve().parent.parent.parent
ALIGNScore_SRC = ROOT / "third_party" / "AlignScore" / "src"
DEFAULT_CKPT = ROOT / "checkpoints" / "AlignScore" / "AlignScore-base.ckpt"


class IssueScorer(Protocol):
    def score(self, issue: str, generated_text: str) -> float: ...


class StubIssueScorer:
    """干跑占位。"""

    def __init__(self, placeholder: float = 0.5):
        self.placeholder = placeholder

    def score(self, issue: str, generated_text: str) -> float:
        _ = (issue, generated_text)
        return self.placeholder


class AlignScoreIssueScorer:
    """AlignScore（RoBERTa NLI）议题对齐，本地 checkpoint，不走生成式 LLM API。"""

    def __init__(
        self,
        *,
        ckpt_path: Optional[str] = None,
        device: str = "cpu",
        batch_size: int = 8,
    ):
        if str(ALIGNScore_SRC) not in sys.path:
            sys.path.insert(0, str(ALIGNScore_SRC))
        from alignscore import AlignScore

        ckpt = Path(ckpt_path or DEFAULT_CKPT)
        if not ckpt.exists():
            raise FileNotFoundError(f"AlignScore checkpoint not found: {ckpt}")

        if device == "cpu":
            dev_arg = "cpu"
        elif device.startswith("cuda"):
            dev_arg = device
        else:
            dev_arg = f"cuda:{device}" if device.isdigit() else device

        self.scorer = AlignScore(
            model="roberta-base",
            batch_size=batch_size,
            device=dev_arg,
            ckpt_path=str(ckpt),
            evaluation_mode="nli_sp",
            verbose=False,
        )

    def score(self, issue: str, generated_text: str) -> float:
        scores = self.scorer.score([issue], [generated_text])
        return float(scores[0])


class LLMIssueScorer:
    """议题维 LLM 兜底（依赖 JudgeClient）。"""

    def __init__(self, judge: JudgeClient, *, max_tokens: int = 1024):
        self.judge = judge
        # R1 系列需较大 max_tokens 以容纳思考链；最终分由 parse_score 截取
        self.max_tokens = max_tokens

    def score(self, issue: str, generated_text: str) -> float:
        prompt = build_issue_prompt({"issue": issue}, generated_text)
        return parse_score(
            self.judge.complete(prompt, temperature=0.0, max_tokens=self.max_tokens)
        )


def create_issue_scorer(
    mode: str = "stub",
    *,
    judge: Optional[JudgeClient] = None,
    alignscore_device: str = "cpu",
    ckpt_path: Optional[str] = None,
    placeholder: float = 0.5,
) -> IssueScorer:
    """
    mode:
      stub       → 占位分
      alignscore → AlignScore（需 checkpoint；device 仅 cpu/cuda，不使用 NPU）
      llm        → 经 JudgeClient 打分（调用待接入）
    """
    key = (mode or "stub").strip().lower()
    if key in {"stub", "dry", "none"}:
        return StubIssueScorer(placeholder=placeholder)
    if key in {"alignscore", "align"}:
        return AlignScoreIssueScorer(ckpt_path=ckpt_path, device=alignscore_device)
    if key in {"llm", "geval"}:
        if judge is None:
            raise ValueError("issue mode=llm 需要传入 judge")
        return LLMIssueScorer(judge)
    raise KeyError(f"未知 issue scorer mode: {mode!r}")
