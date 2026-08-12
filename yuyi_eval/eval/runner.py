"""T-G01-2 评估流水线：读 JSONL → 四维打分 → CSV + summary。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Optional

from yuyi_eval.eval.aggregate import aggregate_metrics
from yuyi_eval.eval.geval import GEvalDimensions
from yuyi_eval.eval.issue import IssueScorer, create_issue_scorer
from yuyi_eval.eval.judge import JudgeClient, create_judge
from yuyi_eval.eval.length import count_chars, score_length
from yuyi_eval.text_utils import strip_reasoning

CSV_FIELDS = [
    "sample_id",
    "baseline",
    "tmodel",
    "judge_tmodel",
    "issue_mode",
    "char_count",
    "length_limit",
    "propaganda_method",
    "stance_label",
    "s_issue",
    "s_method",
    "s_stance",
    "s_length",
    "s_overall",
    "generated_text",
]


def evaluate_sample(
    guidance: dict,
    generated_text: str,
    *,
    geval: GEvalDimensions,
    issue_scorer: IssueScorer,
) -> dict[str, float]:
    text = strip_reasoning(generated_text)
    s_issue = issue_scorer.score(guidance["issue"], text)
    s_method = geval.score_method(guidance, text)
    s_stance = geval.score_stance(guidance, text)
    s_length = score_length(text, guidance["length_limit"])
    s_overall = (s_issue + s_method + s_stance + s_length) / 4.0
    return {
        "s_issue": s_issue,
        "s_method": s_method,
        "s_stance": s_stance,
        "s_length": s_length,
        "s_overall": s_overall,
    }


def _load_existing_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows: list[dict[str, Any]] = []
        for row in reader:
            for key in ("s_issue", "s_method", "s_stance", "s_length", "s_overall", "char_count"):
                if key in row and row[key] != "":
                    row[key] = float(row[key])
            rows.append(row)
        return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def evaluate_file(
    input_jsonl: str,
    output_csv: str,
    *,
    judge_name: str = "stub",
    issue_mode: str = "stub",
    alignscore_device: str = "cpu",
    alignscore_ckpt: Optional[str] = None,
    tau: float = 0.8,
    resume: bool = True,
    stub_score: float = 0.5,
    limit: Optional[int] = None,
    judge_max_tokens: int = 1024,
) -> list[dict[str, Any]]:
    """
    judge_name: stub | r132B | r18B | gpt4o-mini | ...
    issue_mode: stub | alignscore | llm
    limit: 仅评测前 N 条未完成样本（用于冒烟）
    """
    judge: JudgeClient = create_judge(judge_name, placeholder=str(stub_score))
    geval = GEvalDimensions(judge, max_tokens=judge_max_tokens)
    issue_scorer = create_issue_scorer(
        issue_mode,
        judge=judge,
        alignscore_device=alignscore_device,
        ckpt_path=alignscore_ckpt,
        placeholder=stub_score,
    )

    out_path = Path(output_csv)
    existing_rows = _load_existing_csv(out_path) if resume else []
    done_ids = {str(r.get("sample_id", "")) for r in existing_rows}

    new_rows: list[dict[str, Any]] = []
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if limit is not None and len(new_rows) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            sid = str(item.get("sample_id", ""))
            if sid in done_ids:
                continue
            guidance = item["guidance"]
            generated = strip_reasoning(item.get("generated_text", ""))
            scores = evaluate_sample(
                guidance, generated, geval=geval, issue_scorer=issue_scorer
            )
            row = {
                "sample_id": sid,
                "baseline": item.get("baseline", ""),
                "tmodel": item.get("tmodel", ""),
                "judge_tmodel": getattr(judge, "tmodel", judge_name),
                "issue_mode": issue_mode,
                "char_count": count_chars(generated),
                "length_limit": guidance.get("length_limit", ""),
                "propaganda_method": guidance.get("propaganda_method", ""),
                "stance_label": guidance.get("stance", {}).get("label", ""),
                "generated_text": generated,
                **scores,
            }
            new_rows.append(row)
            print(
                f"eval {sid} "
                f"issue={scores['s_issue']:.2f} method={scores['s_method']:.2f} "
                f"stance={scores['s_stance']:.2f} length={scores['s_length']:.1f} "
                f"overall={scores['s_overall']:.3f}",
                flush=True,
            )

    all_rows = existing_rows + new_rows
    _write_csv(out_path, all_rows)

    summary = aggregate_metrics(all_rows, tau=tau)
    summary["judge"] = judge_name
    summary["issue_mode"] = issue_mode
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Summary:", summary)
    print(f"Saved -> {out_path}")
    print(f"Saved -> {summary_path}")
    return all_rows


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="T-G01-2 语义一致性评估（默认 stub 裁判，不调本地 LLM）"
    )
    p.add_argument("--input", required=True, help="含 guidance + generated_text 的 JSONL")
    p.add_argument("--output", required=True, help="输出 CSV 路径")
    p.add_argument(
        "--judge",
        default="stub",
        help="裁判：stub（默认干跑）| r132B | r18B | gpt4o-mini | ...",
    )
    p.add_argument(
        "--issue-mode",
        default="stub",
        choices=["stub", "alignscore", "llm"],
        help="议题维：stub | alignscore | llm",
    )
    p.add_argument(
        "--alignscore-device",
        default="cpu",
        help="AlignScore 设备（cpu / cuda:0 等；不使用 NPU）",
    )
    p.add_argument("--alignscore-ckpt", default=None, help="AlignScore checkpoint 路径")
    p.add_argument("--tau", type=float, default=0.8, help="PassRate 阈值")
    p.add_argument("--stub-score", type=float, default=0.5, help="stub 占位分")
    p.add_argument("--limit", type=int, default=None, help="仅评测前 N 条（冒烟）")
    p.add_argument(
        "--judge-max-tokens",
        type=int,
        default=1024,
        help="裁判生成 max_tokens（R1 思考链需较大）",
    )
    p.add_argument("--no-resume", action="store_true", help="不从已有 CSV 断点续跑")
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    evaluate_file(
        args.input,
        args.output,
        judge_name=args.judge,
        issue_mode=args.issue_mode,
        alignscore_device=args.alignscore_device,
        alignscore_ckpt=args.alignscore_ckpt,
        tau=args.tau,
        resume=not args.no_resume,
        stub_score=args.stub_score,
        limit=args.limit,
        judge_max_tokens=args.judge_max_tokens,
    )


if __name__ == "__main__":
    main()
