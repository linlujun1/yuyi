"""T-G01-2 四维度评测与全集指标汇总."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from yuyi_eval.evaluate.alignscore_eval import AlignScoreEvaluator
from yuyi_eval.evaluate.geval_eval import GEvalJudge, _parse_score
from yuyi_eval.text_utils import strip_reasoning
from yuyi_eval.llm_router import LLMRouter


def count_chars(text: str) -> int:
    return len(str(text).replace(" ", "").replace("\n", ""))


def score_length(generated_text: str, length_limit: str) -> float:
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
    if length_limit == "long":
        if n >= 141:
            return 1.0
        if n >= 120:
            return 0.5
        return 0.0
    raise KeyError(f"unknown length_limit: {length_limit}")


def evaluate_sample(
    guidance: dict,
    generated_text: str,
    *,
    geval: GEvalJudge,
    align: Optional[AlignScoreEvaluator] = None,
) -> dict[str, float]:
    if align is not None:
        s_issue = align.score_issue_consistency(guidance["issue"], generated_text)
    else:
        s_issue = _llm_score_issue(geval, guidance, generated_text)

    s_method = geval.score_method(guidance, generated_text)
    s_stance = geval.score_stance(guidance, generated_text)
    s_len = score_length(generated_text, guidance["length_limit"])
    s_overall = (s_issue + s_method + s_stance + s_len) / 4.0
    return {
        "s_issue": s_issue,
        "s_method": s_method,
        "s_stance": s_stance,
        "s_length": s_len,
        "s_overall": s_overall,
    }


def _llm_score_issue(geval: GEvalJudge, guidance: dict, generated_text: str) -> float:
    from yuyi_eval.evaluate.geval_eval import _parse_score

    prompt = f"""判断生成文本是否围绕指定议题展开、有无明显跑题。

【议题】{guidance['issue']}
【生成文本】
{generated_text}

只输出 0 到 1 之间的小数（1=完全一致，0=完全跑题）。只输出数字。"""
    resp = geval.router.chat(
        geval.judge_tmodel,
        [{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=32,
    )
    return _parse_score(strip_reasoning(resp))


def aggregate_metrics(rows: list[dict[str, Any]], tau: float = 0.8) -> dict[str, float]:
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    dim_cols = ["s_issue", "s_method", "s_stance", "s_length"]
    dim_avg = {c: float(df[c].mean()) for c in dim_cols}
    dim_values = list(dim_avg.values())
    pass_rate = float(
        ((df[dim_cols] >= tau).all(axis=1)).mean()
    )
    return {
        "MacroAvg": float(df["s_overall"].mean()),
        "PassRate": pass_rate,
        "WorstDim": min(dim_values),
        "Gap": max(dim_values) - min(dim_values),
        **{f"DimAvg_{k}": v for k, v in dim_avg.items()},
    }


def evaluate_file(
    input_jsonl: str,
    output_csv: str,
    *,
    judge_tmodel: str = "gpt4o-mini",
    alignscore_device: str = "cpu",
    use_alignscore: bool = True,
    tau: float = 0.8,
    resume: bool = True,
) -> pd.DataFrame:
    router = LLMRouter()
    geval = GEvalJudge(router, judge_tmodel=judge_tmodel)
    align = (
        AlignScoreEvaluator(device=alignscore_device)
        if use_alignscore
        else None
    )

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids: set[str] = set()
    existing_rows: list[dict[str, Any]] = []

    if resume and out_path.exists():
        prev = pd.read_csv(out_path)
        existing_rows = prev.to_dict("records")
        if "sample_id" in prev.columns:
            done_ids = set(prev["sample_id"].astype(str))

    new_rows: list[dict[str, Any]] = []
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            sid = str(item.get("sample_id", ""))
            if sid in done_ids:
                continue
            guidance = item["guidance"]
            generated = strip_reasoning(item.get("generated_text", ""))
            scores = evaluate_sample(guidance, generated, geval=geval, align=align)
            row = {
                "sample_id": sid,
                "baseline": item.get("baseline", ""),
                "tmodel": item.get("tmodel", ""),
                "char_count": count_chars(generated),
                "generated_text": generated,
                **scores,
            }
            new_rows.append(row)
            print(f"eval {sid} overall={scores['s_overall']:.3f}")

    all_rows = existing_rows + new_rows
    df = pd.DataFrame(all_rows)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    summary = aggregate_metrics(all_rows, tau=tau)
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Summary:", summary)
    print(f"Saved -> {out_path}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="T-G01-2 evaluation")
    parser.add_argument("--input", required=True, help="JSONL with guidance + generated_text")
    parser.add_argument("--output", required=True, help="Output CSV")
    parser.add_argument("--judge-tmodel", default="gpt4o-mini")
    parser.add_argument("--alignscore-device", default="cpu")
    parser.add_argument("--no-alignscore", action="store_true")
    parser.add_argument("--tau", type=float, default=0.8)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    evaluate_file(
        args.input,
        args.output,
        judge_tmodel=args.judge_tmodel,
        alignscore_device=args.alignscore_device,
        use_alignscore=not args.no_alignscore,
        tau=args.tau,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
