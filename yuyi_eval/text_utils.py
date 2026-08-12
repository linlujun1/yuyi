"""LLM 输出后处理."""

from __future__ import annotations

_REDACTED_END = "</think>"


def strip_reasoning(text: str) -> str:
    """去掉 DeepSeek-R1 等模型的思考链，保留最终正文."""
    if _REDACTED_END in text:
        text = text.rsplit(_REDACTED_END, 1)[-1]
    lower = text.lower()
    idx = lower.rfind("/think>")
    if idx != -1:
        text = text[idx + len("/think>") :]
    return text.strip()
