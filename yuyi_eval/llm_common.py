from __future__ import annotations

import re


QWEN3_NO_THINK_MODELS = {
    "Qwen3-32B",
}


def add_no_think_for_qwen3(
    model: str,
    prompt: str,
) -> str:
    if model in QWEN3_NO_THINK_MODELS:
        return "/no_think\n" + prompt

    return prompt


def clean_thinking_text(text: str) -> str:
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]

    text = re.sub(
        r"<think>.*?(</think>|$)",
        "",
        text,
        flags=re.S,
    )

    return text.strip()


def message_text(message: dict) -> str | None:
    return (
        message.get("content")
        or message.get("reasoning")
        or message.get("reasoning_content")
    )
