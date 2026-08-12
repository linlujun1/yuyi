"""
裁判客户端接口：支持 stub / 任意 config/models.yaml 中的 tmodel 切换。

默认裁判模型 tmodel = r132B（DeepSeek-R1-Distill-Qwen-32B）。
"""

from __future__ import annotations

from typing import Optional, Protocol

from yuyi_eval.llm_router import LLMRouter


class JudgeClient(Protocol):
    """G-Eval 裁判：输入 prompt，返回原始文本（再由 parse_score 解析）。"""

    tmodel: str

    def complete(self, prompt: str, *, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        ...


class StubJudgeClient:
    """干跑 / 流程联调用：不访问任何模型，固定返回占位分数文本。"""

    def __init__(self, tmodel: str = "stub", placeholder: str = "0.5"):
        self.tmodel = tmodel
        self.placeholder = placeholder

    def complete(self, prompt: str, *, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        _ = (prompt, temperature, max_tokens)
        return self.placeholder


class LLMJudgeClient:
    """
    经 LLMRouter 调用本地 / 云端 OpenAI 兼容接口。

    切换模型：传入 config/models.yaml 中的 tmodel，例如
      - r132B   DeepSeek-R1-Distill-Qwen-32B（默认）
      - r18B    DeepSeek-R1-Distill-Llama-8B
      - gpt4o-mini
    """

    def __init__(self, tmodel: str = "r132B", router: Optional[LLMRouter] = None):
        self.tmodel = tmodel
        self._router = router or LLMRouter()

    def complete(self, prompt: str, *, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        # R1 蒸馏模型常先输出思考链；默认 1024，调用方仍可覆盖
        return self._router.chat(
            self.tmodel,
            [{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )


def create_judge(
    name: str = "stub",
    *,
    placeholder: str = "0.5",
    router: Optional[LLMRouter] = None,
) -> JudgeClient:
    """
    工厂：按名称创建裁判客户端。

    name:
      stub          → StubJudgeClient
      r132B / r18B / gpt4o-mini / ... → LLMJudgeClient(tmodel=name)
    """
    key = (name or "stub").strip().lower()
    if key in {"stub", "dry", "dry-run", "none"}:
        return StubJudgeClient(tmodel="stub", placeholder=placeholder)
    return LLMJudgeClient(tmodel=name, router=router)
