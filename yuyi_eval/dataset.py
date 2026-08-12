"""T-G01-2 数据集：议题翻译与引导条件采样."""

from __future__ import annotations

import csv
import json
import random
import re
from pathlib import Path
from typing import Any, Optional

from yuyi_eval.llm_router import LLMRouter
from yuyi_eval.prompt_builder import PROPAGANDA_PROMPTS

STANCE_LABELS = ("支持", "中立", "反对")
LENGTH_LIMITS = ("short", "medium", "long")

DEFAULT_TOPIC_CSV = Path(__file__).resolve().parent.parent / "认知BC子项目" / "topic.csv"
DEFAULT_TOPICS_JSON = Path(__file__).resolve().parent.parent / "data" / "议题.json"


def load_topics(path: str | Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("议题.json 应为数组")
    return data


def _read_topic_csv(csv_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            issue_en = (row.get("issue") or "").strip()
            if not issue_en:
                continue
            rows.append(
                {
                    "id": int(row["i"]),
                    "issue_en": issue_en,
                    "type": (row.get("type") or "").strip(),
                }
            )
    return rows


def _load_translated_map(output_path: Path) -> dict[int, dict[str, Any]]:
    if not output_path.exists():
        return {}
    items = load_topics(output_path)
    return {int(x["id"]): x for x in items if "id" in x}


def _parse_translation_batch(text: str, expected_ids: list[int]) -> dict[int, str]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None

    out: dict[int, str] = {}
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict) and "id" in item and "issue" in item:
                out[int(item["id"])] = str(item["issue"]).strip()
    elif isinstance(payload, dict):
        for k, v in payload.items():
            try:
                out[int(k)] = str(v).strip()
            except ValueError:
                continue

    if len(out) >= len(expected_ids):
        return out

    # 回退：逐行 "id: 中文"
    for line in text.splitlines():
        m = re.match(r"^\s*(\d+)\s*[:：]\s*(.+?)\s*$", line)
        if m:
            out[int(m.group(1))] = m.group(2).strip()
    return out


def translate_topics_with_llm(
    csv_path: str | Path = DEFAULT_TOPIC_CSV,
    output_path: str | Path = DEFAULT_TOPICS_JSON,
    *,
    tmodel: str = "gpt4o-mini",
    batch_size: int = 10,
    limit: Optional[int] = None,
    router: Optional[LLMRouter] = None,
) -> list[dict[str, Any]]:
    """用 LLM 将 topic.csv 译为中文议题.json（支持断点续翻）."""
    csv_path = Path(csv_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    router = router or LLMRouter()
    rows = _read_topic_csv(csv_path)
    if limit is not None:
        rows = rows[:limit]

    done = _load_translated_map(output_path)
    pending = [r for r in rows if r["id"] not in done]

    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        ids = [r["id"] for r in batch]
        lines = "\n".join(f'{r["id"]}. {r["issue_en"]}' for r in batch)
        prompt = f"""请将下列英文公共议题逐条翻译为中文，用于新闻评论场景。

要求：
1. 必须使用自然、地道的中文表述，符合中文媒体/舆论场语境
2. 保留原意，允许必要意译，但禁止生硬直译和机翻腔
3. 每条输出为简洁的议题命题句（20–80字为宜）
4. 只输出 JSON 数组，格式：[{{"id": 1, "issue": "中文议题"}}]

英文议题：
{lines}"""

        resp = router.chat(
            tmodel,
            [{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2048,
        )
        translated = _parse_translation_batch(resp, ids)
        for row in batch:
            zh = translated.get(row["id"])
            if not zh:
                raise RuntimeError(f"翻译失败，缺少 id={row['id']}，LLM 返回:\n{resp}")
            done[row["id"]] = {
                "id": row["id"],
                "issue_en": row["issue_en"],
                "issue": zh,
                "type": row["type"],
            }
        _save_topics(output_path, done)
        print(f"translated {min(start + batch_size, len(pending))}/{len(pending)} new topics")

    result = [done[r["id"]] for r in rows if r["id"] in done]
    _save_topics(output_path, {item["id"]: item for item in result})
    print(f"Saved {len(result)} topics -> {output_path}")
    return result


def _save_topics(output_path: Path, items: dict[int, dict[str, Any]]) -> None:
    ordered = [items[k] for k in sorted(items)]
    output_path.write_text(
        json.dumps(ordered, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sample_stance_target(issue: str, label: str) -> str:
    core = issue.strip().rstrip("。．.")
    if len(core) > 48:
        cut = core[:48]
        for sep in ("，", "、", "：", "；" ):
            idx = cut.rfind(sep)
            if idx > 12:
                core = cut[:idx]
                break
        else:
            core = cut + "…"
    if label == "支持":
        return f"{core}的相关做法或方向"
    if label == "反对":
        return f"{core}的相关做法或现象"
    return f"{core}的相关情况"


def sample_guidance_dataset(
    topics_json: str | Path,
    output_jsonl: str | Path,
    *,
    n_samples: int,
    seed: int = 42,
    prefix: str = "tg012",
) -> int:
    """从议题.json 随机采样引导条件，输出实验 JSONL."""
    topics = load_topics(topics_json)
    if not topics:
        raise ValueError("议题.json 为空，请先运行 translate-topics")

    rng = random.Random(seed)
    methods = list(PROPAGANDA_PROMPTS.keys())
    out_path = Path(output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    for i in range(1, n_samples + 1):
        topic = rng.choice(topics)
        label = rng.choice(STANCE_LABELS)
        guidance = {
            "issue": topic["issue"],
            "propaganda_method": rng.choice(methods),
            "stance": {
                "label": label,
                "target": _sample_stance_target(topic["issue"], label),
            },
            "length_limit": rng.choice(LENGTH_LIMITS),
        }
        item = {
            "sample_id": f"{prefix}_{i:04d}",
            "topic_id": topic["id"],
            "guidance": guidance,
        }
        lines.append(json.dumps(item, ensure_ascii=False))

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Sampled {n_samples} -> {out_path}")
    return n_samples
