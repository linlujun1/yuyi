"""T-G01-2 被测生成系统接口（三基线）."""

from __future__ import annotations

from typing import Optional, Protocol

from yuyi_eval.llm_router import LLMRouter, load_models_registry
from yuyi_eval.prompt_builder import build_generation_prompt
from yuyi_eval.text_utils import strip_reasoning


class GenerationBackend(Protocol):
    name: str

    def generate(self, guidance: dict) -> str: ...


class DirectLLMBackend:
    name = "direct"

    def __init__(self, router: LLMRouter, tmodel: str):
        self.router = router
        self.tmodel = tmodel

    def generate(self, guidance: dict) -> str:
        prompt = build_generation_prompt(guidance)
        return strip_reasoning(
            self.router.chat(
                self.tmodel,
                [{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1536,
            )
        )


class MultiAgentBackend:
    """Planner → Writer → Critic 三阶段；各角色 tmodel 见 config/models.yaml."""

    name = "multi_agent"

    def __init__(self, router: LLMRouter, roles: Optional[dict[str, str]] = None):
        self.router = router
        if roles is None:
            reg = load_models_registry(router.config_path)
            roles = reg.get("baselines", {}).get("multi_agent", {}).get("roles", {})
        if not roles:
            raise ValueError("multi_agent 需要 roles 配置（config/models.yaml baselines.multi_agent.roles）")
        self.roles = roles

    def generate(self, guidance: dict) -> str:
        user_prompt = build_generation_prompt(guidance)
        plan = self.router.chat(
            self.roles["planner"],
            [
                {
                    "role": "system",
                    "content": "你是写作策划。根据用户要求，列出 3 条简要写作要点，不要写正文。",
                },
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=256,
        )
        draft = strip_reasoning(
            self.router.chat(
                self.roles["writer"],
                [
                    {
                        "role": "system",
                        "content": f"按以下策划要点写作，严格遵守用户给出的议题、修辞、立场与字数要求。\n\n{plan}",
                    },
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=512,
            )
        )
        final = strip_reasoning(
            self.router.chat(
                self.roles.get("critic", self.roles["writer"]),
                [
                    {
                        "role": "system",
                        "content": "你是编辑。在不改变立场与修辞手法的前提下，润色下列草稿使其更符合字数与议题要求。只输出最终正文。",
                    },
                    {"role": "user", "content": f"{user_prompt}\n\n【草稿】\n{draft}"},
                ],
                temperature=0.5,
                max_tokens=512,
            )
        )
        return final


class StructuredSignalBackend:
    """基线 3：结构化信号引导（接口占位，待毕设方法接入）."""

    name = "structured_signal"

    def generate(self, guidance: dict) -> str:
        raise NotImplementedError(
            "structured_signal 基线尚未实现；请在 StructuredSignalBackend.generate 中接入毕设方法。"
        )


def make_backend(
    baseline: str,
    router: LLMRouter,
    *,
    tmodel: Optional[str] = None,
    roles: Optional[dict[str, str]] = None,
) -> GenerationBackend:
    if baseline == "direct":
        if not tmodel:
            raise ValueError("direct 基线需要 --tmodel")
        return DirectLLMBackend(router, tmodel)
    if baseline == "multi_agent":
        return MultiAgentBackend(router, roles=roles)
    if baseline == "structured_signal":
        return StructuredSignalBackend()
    raise ValueError(f"未知 baseline: {baseline!r}，可选: direct, multi_agent, structured_signal")
