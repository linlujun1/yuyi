from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from openai import OpenAI


def clean_thinking_text(text: str) -> str:
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    text = re.sub(r"<think>.*?(</think>|$)", "", text, flags=re.S)
    return text.strip()


@dataclass
class LLMResult:
    text: str
    usage: dict[str, Any] | None = None


class PlatformLLM:
    def __init__(self):
        self.base_url = os.environ["OPENAI_BASE_URL"].rstrip("/")
        self.api_key = os.environ["OPENAI_API_KEY"]
        self.model = os.environ["LLM_MODEL"]
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def chat(self, system: str, user: str, temperature: float = 0.0, max_tokens: int = 512) -> LLMResult:
        if self.model.lower().startswith("qwen3"):
            system = "/no_think\n" + system

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        msg = resp.choices[0].message
        text = (
            getattr(msg, "content", None)
            or getattr(msg, "reasoning_content", None)
            or ""
        )

        usage = resp.usage.model_dump() if getattr(resp, "usage", None) else None
        return LLMResult(clean_thinking_text(text), usage)
