"""T-G01-2 批量实验入口：数据 / 生成 / 评测 / 流水线."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

from yuyi_eval.backends import make_backend
from yuyi_eval.dataset import (
    DEFAULT_TOPICS_JSON,
    sample_guidance_dataset,
)
from yuyi_eval.evaluate.tg012_eval import evaluate_file
from yuyi_eval.llm_router import LLMRouter, load_models_registry


def _load_done_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    done: set[str] = set()
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            sid = item.get("sample_id")
            if sid:
                done.add(str(sid))
    return done


def run_generation(
    input_jsonl: str,
    output_jsonl: str,
    *,
    baseline: str,
    tmodel: str | None = None,
    resume: bool = True,
    retries: int = 2,
) -> int:
    router = LLMRouter()
    backend = make_backend(baseline, router, tmodel=tmodel)

    in_path = Path(input_jsonl)
    out_path = Path(output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = _load_done_ids(out_path) if resume else set()

    count = 0
    mode = "a" if resume and out_path.exists() else "w"
    with in_path.open("r", encoding="utf-8") as fin, out_path.open(mode, encoding="utf-8") as fout:
        for i, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            sid = str(item.get("sample_id", i))
            if sid in done_ids:
                continue

            guidance = item["guidance"]
            last_err: Optional[Exception] = None
            for attempt in range(retries + 1):
                try:
                    text = backend.generate(guidance)
                    item["generated_text"] = text
                    item["baseline"] = baseline
                    if tmodel:
                        item["tmodel"] = tmodel
                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                    fout.flush()
                    count += 1
                    print(f"[{i}] {sid} done")
                    last_err = None
                    break
                except Exception as exc:
                    last_err = exc
                    print(f"[{i}] {sid} attempt {attempt + 1} failed: {exc}")
            if last_err is not None:
                raise last_err

    print(f"Generated {count} new samples -> {out_path}")
    return count


def run_pipeline(
    *,
    n_samples: int,
    seed: int,
    baseline: str,
    tmodel: str | None,
    work_dir: str,
    judge_tmodel: str,
    skip_translate: bool = False,
    skip_sample: bool = False,
    skip_generate: bool = False,
    skip_evaluate: bool = False,
    topics_json: str = str(DEFAULT_TOPICS_JSON),
) -> None:
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    samples_path = work / "samples.jsonl"
    gen_path = work / f"generated_{baseline}{f'_{tmodel}' if tmodel else ''}.jsonl"
    eval_path = work / f"metrics_{baseline}{f'_{tmodel}' if tmodel else ''}.csv"

    if not skip_translate and not Path(topics_json).exists():
        raise FileNotFoundError(
            f"议题文件不存在: {topics_json}（请先准备 data/议题.json）"
        )

    if not skip_sample:
        sample_guidance_dataset(topics_json, samples_path, n_samples=n_samples, seed=seed)

    if not skip_generate:
        run_generation(
            str(samples_path),
            str(gen_path),
            baseline=baseline,
            tmodel=tmodel,
            resume=True,
        )

    if not skip_evaluate:
        evaluate_file(
            str(gen_path),
            str(eval_path),
            judge_tmodel=judge_tmodel,
            resume=True,
        )


def run_matrix(work_dir: str, *, seed: int, n_samples: int, judge_tmodel: str) -> None:
    reg = load_models_registry()
    jobs: list[dict[str, Any]] = reg.get("experiments", [])
    if not jobs:
        raise ValueError("config/models.yaml 中 experiments 为空")

    for job in jobs:
        baseline = job["baseline"]
        tmodel = job.get("tmodel")
        subdir = Path(work_dir) / f"{baseline}{f'_{tmodel}' if tmodel else ''}"
        print(f"\n=== matrix job: {baseline} {tmodel or ''} ===")
        run_pipeline(
            n_samples=n_samples,
            seed=seed,
            baseline=baseline,
            tmodel=tmodel,
            work_dir=str(subdir),
            judge_tmodel=judge_tmodel,
            skip_translate=True,
            skip_sample=False,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="T-G01-2 实验")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sm = sub.add_parser("sample", help="从议题.json 采样引导条件 JSONL")
    p_sm.add_argument("--topics", default=str(DEFAULT_TOPICS_JSON))
    p_sm.add_argument("--output", required=True)
    p_sm.add_argument("-n", "--num-samples", type=int, required=True)
    p_sm.add_argument("--seed", type=int, default=42)

    p_gen = sub.add_parser("generate", help="批量生成")
    p_gen.add_argument("--input", required=True)
    p_gen.add_argument("--output", required=True)
    p_gen.add_argument("--baseline", required=True, choices=["direct", "multi_agent", "structured_signal"])
    p_gen.add_argument("--tmodel", default=None)
    p_gen.add_argument("--no-resume", action="store_true")
    p_gen.add_argument("--retries", type=int, default=2)

    p_ev = sub.add_parser("evaluate", help="四维度评测")
    p_ev.add_argument("--input", required=True)
    p_ev.add_argument("--output", required=True)
    p_ev.add_argument("--judge-tmodel", default="gpt4o-mini")
    p_ev.add_argument("--alignscore-device", default="cpu")
    p_ev.add_argument("--no-alignscore", action="store_true")
    p_ev.add_argument("--no-resume", action="store_true")

    p_pl = sub.add_parser("pipeline", help="采样 -> 生成 -> 评测 一条龙")
    p_pl.add_argument("--work-dir", required=True)
    p_pl.add_argument("-n", "--num-samples", type=int, default=5)
    p_pl.add_argument("--seed", type=int, default=42)
    p_pl.add_argument("--baseline", required=True, choices=["direct", "multi_agent", "structured_signal"])
    p_pl.add_argument("--tmodel", default=None)
    p_pl.add_argument("--judge-tmodel", default="gpt4o-mini")
    p_pl.add_argument("--topics", default=str(DEFAULT_TOPICS_JSON))
    p_pl.add_argument("--skip-translate", action="store_true")
    p_pl.add_argument("--skip-sample", action="store_true")
    p_pl.add_argument("--skip-generate", action="store_true")
    p_pl.add_argument("--skip-evaluate", action="store_true")

    p_mx = sub.add_parser("run-matrix", help="按 models.yaml experiments 批量跑实验")
    p_mx.add_argument("--work-dir", required=True)
    p_mx.add_argument("-n", "--num-samples", type=int, default=5)
    p_mx.add_argument("--seed", type=int, default=42)
    p_mx.add_argument("--judge-tmodel", default="gpt4o-mini")

    sub.add_parser("list-models")

    p_mindie = sub.add_parser(
        "render-mindie-config",
        help="根据 runtime 渲染 MindIE JSON（需先 plan/start）",
    )
    p_mindie.add_argument("--tmodel", required=True)
    p_mindie.add_argument("--output", required=True)

    args = parser.parse_args()

    if args.cmd == "list-models":
        for name in LLMRouter().list_tmodels():
            print(name)
        return

    if args.cmd == "render-mindie-config":
        LLMRouter().dump_mindie_config(args.tmodel, args.output)
        print(f"Wrote {args.output}")
        return

    if args.cmd == "sample":
        sample_guidance_dataset(
            args.topics,
            args.output,
            n_samples=args.num_samples,
            seed=args.seed,
        )
        return

    if args.cmd == "generate":
        run_generation(
            args.input,
            args.output,
            baseline=args.baseline,
            tmodel=args.tmodel,
            resume=not args.no_resume,
            retries=args.retries,
        )
        return

    if args.cmd == "evaluate":
        evaluate_file(
            args.input,
            args.output,
            judge_tmodel=args.judge_tmodel,
            alignscore_device=args.alignscore_device,
            use_alignscore=not args.no_alignscore,
            resume=not args.no_resume,
        )
        return

    if args.cmd == "pipeline":
        run_pipeline(
            n_samples=args.num_samples,
            seed=args.seed,
            baseline=args.baseline,
            tmodel=args.tmodel,
            work_dir=args.work_dir,
            judge_tmodel=args.judge_tmodel,
            skip_translate=args.skip_translate,
            skip_sample=args.skip_sample,
            skip_generate=args.skip_generate,
            skip_evaluate=args.skip_evaluate,
            topics_json=args.topics,
        )
        return

    if args.cmd == "run-matrix":
        run_matrix(
            args.work_dir,
            seed=args.seed,
            n_samples=args.num_samples,
            judge_tmodel=args.judge_tmodel,
        )


if __name__ == "__main__":
    main()
