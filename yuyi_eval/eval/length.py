"""字数一致性：纯规则，不依赖模型。"""

from __future__ import annotations

LENGTH_RANGES: dict[str, tuple[int | None, int | None]] = {
    "short": (None, 50),
    "medium": (51, 140),
    "long": (141, None),
}


def count_chars(text: str) -> int:
    """计数字符，不含空白。"""
    return len(str(text).replace(" ", "").replace("\n", "").replace("\t", ""))


def score_length(generated_text: str, length_limit: str) -> float:
    """
    落在引导档位区间 → 1.0；相邻档轻微越界 → 0.5；明显越界 → 0.0。
    """
    if length_limit not in LENGTH_RANGES:
        raise KeyError(f"unknown length_limit: {length_limit!r}")

    n = count_chars(generated_text)
    if length_limit == "short":
        if n <= 50:
            return 1.0
        if n <= 70:
            return 0.5
        return 0.0
    if length_limit == "medium":
        if 51 <= n <= 140:
            return 1.0
        if n <= 50 or n <= 160:
            return 0.5
        return 0.0
    # long
    if n >= 141:
        return 1.0
    if n >= 120:
        return 0.5
    return 0.0
