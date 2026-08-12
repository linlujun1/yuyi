"""从议题.json 采样引导条件，调用指定 tmodel 生成评论。

输出: data/gencomment/
  python -m yuyi_eval.gen.generate_comments --gen_num 50 --tmodel r18B
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from yuyi_eval.backends import DirectLLMBackend
from yuyi_eval.dataset import (
    DEFAULT_TOPICS_JSON,
    LENGTH_LIMITS,
    STANCE_LABELS,
    _sample_stance_target,
    load_topics,
)
from yuyi_eval.llm_router import LLMRouter, get_endpoint
from yuyi_eval.prompt_builder import PROPAGANDA_PROMPTS

DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "gencomment"


def _build_sample(topic: dict[str, Any], *, sample_id: str, rng: random.Random) -> dict[str, Any]:
    label = rng.choice(STANCE_LABELS)
    issue = topic["issue"]
    return {
        "sample_id": sample_id,
        "topic_id": topic["id"],
        "guidance": {
            "issue": issue,
            "propaganda_method": rng.choice(list(PROPAGANDA_PROMPTS.keys())),
            "stance": {"label": label, "target": _sample_stance_target(issue, label)},
            "length_limit": rng.choice(LENGTH_LIMITS),
        },
    }


def _load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sid = json.loads(line).get("sample_id")
            if sid:
                done.add(str(sid))
    return done


def generate_comments(
    *,
    gen_num: int,
    tmodel: str,
    topics_json: str | Path = DEFAULT_TOPICS_JSON,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    seed: int = 42,
    resume: bool = True,
    retries: int = 2,
    prefix: Optional[str] = None,
) -> Path:
    if gen_num <= 0:
        raise ValueError("gen_num 必须 > 0")
    ep = get_endpoint(tmodel)
    topics = load_topics(topics_json)
    if not topics:
        raise ValueError(f"议题为空: {topics_json}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = prefix or tmodel
    out_path = out_dir / f"comments_{tag}.jsonl"
    router = LLMRouter()
    backend = DirectLLMBackend(router, tmodel)
    rng = random.Random(seed)
    done_ids = _load_done_ids(out_path) if resume else set()
    mode = "a" if resume and out_path.exists() else "w"
    written = 0
    with out_path.open(mode, encoding="utf-8") as fout:
        for i in range(1, gen_num + 1):
            sid = f"{tag}_{i:04d}"
            if sid in done_ids:
                continue
            item = _build_sample(rng.choice(topics), sample_id=sid, rng=rng)
            last_err: Optional[Exception] = None
            for attempt in range(retries + 1):
                try:
                    text = backend.generate(item["guidance"])
                    if not text or not text.strip():
                        raise RuntimeError("模型返回空文本")
                    item["generated_text"] = text
                    item["tmodel"] = tmodel
                    item["display_name"] = ep.display_name
                    item["created_at"] = datetime.now(timezone.utc).isoformat()
                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                    fout.flush()
                    written += 1
                    print(f"[{written}/{gen_num}] {sid} ok ({len(text)} chars)")
                    last_err = None
                    break
                except Exception as exc:
                    last_err = exc
                    print(f"[{i}/{gen_num}] {sid} attempt {attempt + 1} failed: {exc}")
            if last_err is not None:
                raise last_err
    print(f"Wrote {written} new comments -> {out_path} (tmodel={tmodel})")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="由议题调用 LLM 生成评论")
    p.add_argument("--gen_num", type=int, required=True)
    p.add_argument("--tmodel", type=str, required=True)
    p.add_argument("--topics", default=str(DEFAULT_TOPICS_JSON))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--prefix", default=None)
    a = p.parse_args()
    generate_comments(
        gen_num=a.gen_num,
        tmodel=a.tmodel,
        topics_json=a.topics,
        out_dir=a.out_dir,
        seed=a.seed,
        resume=not a.no_resume,
        retries=a.retries,
        prefix=a.prefix,
    )


if __name__ == "__main__":
    main()
