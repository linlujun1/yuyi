from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from llm_service.model_service import MODELS, ModelService


DEFAULT_EVALUATOR = "DeepSeek-R1-Distill-Qwen-32B"
EVAL_METHODS = [
    "direct_llm",
    "multi_agent",
]

def load_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if line:
                records.append(
                    json.loads(line)
                )

    return records


def load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()

    done = set()

    for record in load_jsonl(path):
        done.add(record["sample_id"])

    return done


def get_target(
    guidance: dict,
    with_target: bool,
) -> str:
    if not with_target:
        return guidance["issue"]

    target = guidance["stance"].get("target")

    if not target:
        raise ValueError(
            "开启 --with-target，"
            "但样本缺少 guidance.stance.target"
        )

    return target


def score_length(
    text: str,
    length_limit: str,
) -> float:
    length = len(text.strip())

    if length_limit == "short":
        return 1.0 if length <= 50 else 0.0

    if length_limit == "medium":
        return (
            1.0
            if 51 <= length <= 140
            else 0.0
        )

    if length_limit == "long":
        return 1.0 if length >= 141 else 0.0

    raise ValueError(
        f"未知长度档位: {length_limit}"
    )


def build_eval_prompt(
    guidance: dict,
    generated_text: str,
    with_target: bool,
) -> str:
    target = get_target(
        guidance,
        with_target,
    )

    return f"""
你是一个文本语义一致性评估器。

请评估生成文本是否符合给定要求。

只评估以下三个维度：

1. issue
生成文本是否围绕指定议题展开。
1 = 一致
0.5 = 基本相关，但存在明显偏题、遗漏或轻度偷换
0 = 主要讨论其他议题

2. method
生成文本是否主要体现指定宣传手段。
1 = 指定宣传手段是主要表达策略
0.5 = 有体现，但不明显或不是主要策略
0 = 基本没有体现指定宣传手段

3. stance
生成文本对指定目标的实际态度是否符合要求。
1 = 立场一致
0.5 = 立场不明确、混合或部分一致
0 = 立场明显相反

议题：
{guidance["issue"]}

指定宣传手段：
{guidance["propaganda_method"]}

指定立场：
{guidance["stance"]["label"]}

立场判断目标：
{target}

生成文本：
{generated_text}

只能输出 JSON，不要输出解释，不要输出 Markdown。

格式必须严格为：

{{
  "issue": 0,
  "method": 0,
  "stance": 0
}}

每个分数只能是：
0、0.5、1
""".strip()


def chat_completion(
    base_url: str,
    model: str,
    prompt: str,
) -> tuple[str, dict]:
    body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": 0,
        "max_tokens": 1024,
    }

    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(
            body,
            ensure_ascii=False,
        ).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(
        request,
        timeout=300,
    ) as response:
        result = json.loads(
            response.read().decode("utf-8")
        )

    content = (
        result["choices"][0]["message"]["content"]
    )

    return content, result.get("usage", {})


def parse_scores(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(
            f"评估模型没有返回 JSON: {text}"
        )

    scores = json.loads(
        text[start:end + 1]
    )

    result = {}

    for key in [
        "issue",
        "method",
        "stance",
    ]:
        if key not in scores:
            raise ValueError(
                f"评估结果缺少 {key}: {text}"
            )

        value = float(scores[key])

        if value not in {
            0.0,
            0.5,
            1.0,
        }:
            raise ValueError(
                f"{key} 分数非法: {value}"
            )

        result[key] = value

    return result

def evaluate_semantic_scores(
    guidance: dict,
    generated_text: str,
    evaluator: str,
    base_url: str,
    with_target: bool,
    method: str,
) -> tuple[dict, dict, dict | None]:
    if method == "direct_llm":
        prompt = build_eval_prompt(
            guidance=guidance,
            generated_text=generated_text,
            with_target=with_target,
        )

        response_text, usage = chat_completion(
            base_url=base_url,
            model=evaluator,
            prompt=prompt,
        )

        scores = parse_scores(
            response_text
        )

        return scores, usage,None

    if method == "multi_agent":
        roles = [
            "semantic_reviewer",
            "rhetoric_reviewer",
            "adversarial_reviewer",
        ]

        agent_results = []
        usage = {
            "agents": {}
        }

        for role in roles:
            prompt = build_multi_agent_prompt(
                guidance=guidance,
                generated_text=generated_text,
                with_target=with_target,
                role=role,
            )

            response_text, agent_usage = (
                chat_completion(
                    base_url=base_url,
                    model=evaluator,
                    prompt=prompt,
                )
            )

            scores = parse_scores(
                response_text
            )

            agent_results.append(
                {
                    "role": role,
                    "scores": scores,
                    "text": response_text,
                }
            )

            usage["agents"][role] = (
                agent_usage
            )

        final_prompt = build_adjudicator_prompt(
            guidance=guidance,
            generated_text=generated_text,
            with_target=with_target,
            agent_results=agent_results,
        )

        response_text, final_usage = (
            chat_completion(
                base_url=base_url,
                model=evaluator,
                prompt=final_prompt,
            )
        )

        final_scores = parse_scores(
            response_text
        )

        usage["adjudicator"] = (
            final_usage
        )

        return final_scores, usage, {
            "agents": agent_results["agents"],
            "adjudicator": {
                "text": response_text
            }
        }

    raise ValueError(
        f"未知评估方法: {method}"
    )

def build_multi_agent_prompt(
    guidance: dict,
    generated_text: str,
    with_target: bool,
    role: str,
) -> str:
    target = get_target(
        guidance,
        with_target,
    )

    role_instructions = {
        "semantic_reviewer": (
            "你是语义一致性审查员。"
            "重点检查文本是否真正围绕指定议题，"
            "是否存在偏题、偷换概念，同时也完整评价宣传手段和立场。"
        ),
        "rhetoric_reviewer": (
            "你是宣传与修辞策略审查员。"
            "重点判断指定宣传手段是否真正构成文本的主要表达策略，"
            "同时也完整评价议题和立场。"
        ),
        "adversarial_reviewer": (
            "你是严格的反方验证审查员。"
            "主动寻找文本与要求不一致的证据，"
            "防止因为表面关键词相似而给出过高分数，"
            "并完整评价议题、宣传手段和立场。"
        ),
    }

    if role not in role_instructions:
        raise ValueError(
            f"未知 agent 角色: {role}"
        )

    return f"""
{role_instructions[role]}

请独立评估以下生成文本。

评分维度：

1. issue
1 = 与指定议题一致
0.5 = 基本相关，但存在明显偏题、遗漏或轻度偷换
0 = 主要讨论其他议题

2. method
1 = 指定宣传手段是主要表达策略
0.5 = 有体现，但不明显或不是主要策略
0 = 基本没有体现指定宣传手段

3. stance
1 = 对指定目标的实际态度与指定立场一致
0.5 = 立场不明确、混合或部分一致
0 = 立场明显相反

指定议题：
{guidance["issue"]}

指定宣传手段：
{guidance["propaganda_method"]}

指定立场：
{guidance["stance"]["label"]}

立场判断目标：
{target}

生成文本：
{generated_text}

只能输出 JSON，不要输出解释，不要输出 Markdown。

格式：

{{
  "issue": 0,
  "method": 0,
  "stance": 0
}}

每个分数只能是：
0、0.5、1
""".strip()

def build_adjudicator_prompt(
    guidance: dict,
    generated_text: str,
    with_target: bool,
    agent_results: list[dict],
) -> str:
    target = get_target(
        guidance,
        with_target,
    )

    reviews = json.dumps(
        agent_results,
        ensure_ascii=False,
        indent=2,
    )

    return f"""
你是多智能体评估系统的最终裁决员。

三名独立审查员已经分别评价了同一文本。
你需要结合原始要求、生成文本以及三名审查员的意见，
给出最终评分。

不要简单多数投票。
如果审查员意见冲突，应根据原始文本和评价标准自行裁决。

评分维度：

issue：
1 = 与指定议题一致
0.5 = 基本相关，但存在明显偏题、遗漏或轻度偷换
0 = 主要讨论其他议题

method：
1 = 指定宣传手段是主要表达策略
0.5 = 有体现，但不明显或不是主要策略
0 = 基本没有体现指定宣传手段

stance：
1 = 对指定目标的实际态度与指定立场一致
0.5 = 立场不明确、混合或部分一致
0 = 立场明显相反

指定议题：
{guidance["issue"]}

指定宣传手段：
{guidance["propaganda_method"]}

指定立场：
{guidance["stance"]["label"]}

立场判断目标：
{target}

生成文本：
{generated_text}

三名审查员结果：
{reviews}

只能输出最终 JSON，不要输出解释，不要输出 Markdown。

格式：

{{
  "issue": 0,
  "method": 0,
  "stance": 0
}}

每个分数只能是：
0、0.5、1
""".strip()
def evaluate_one(
    case: dict,
    generation: dict,
    evaluator: str,
    base_url: str,
    with_target: bool,
    method: str,
) -> dict:
    guidance = case["guidance"]
    generated_text = generation["generated_text"]

    scores, usage, agent_results = evaluate_semantic_scores(
        guidance=guidance,
        generated_text=generated_text,
        evaluator=evaluator,
        base_url=base_url,
        with_target=with_target,
        method=method,
    )
    scores["length"] = score_length(
        generated_text,
        guidance["length_limit"],
    )

    scores["overall"] = sum(
        [
            scores["issue"],
            scores["method"],
            scores["stance"],
            scores["length"],
        ]
    ) / 4

    return {
        "sample_id": case["sample_id"],
        "source_backend": generation.get(
            "backend"
        ),
        "source_model": generation.get(
            "model"
        ),
        "eval_method": method,
        "evaluator_model": evaluator,
        "with_target": with_target,
        "scores": scores,
        "char_count": len(
            generated_text.strip()
        ),
        "usage": usage,
        "_agent_results": agent_results,
    }

def evaluate_comments(
    cases_path: Path,
    input_path: Path,
    evaluator: str,
    with_target: bool,
    limit: int | None,
    method: str,
) -> Path:
    cases = load_jsonl(
        cases_path
    )

    generations = load_jsonl(
        input_path
    )

    cases_by_id = {
        case["sample_id"]: case
        for case in cases
    }

    if limit is not None:
        generations = generations[:limit]

    source_model = (
        generations[0].get("model", "unknown")
        if generations
        else "unknown"
    )

    mode = (
        "target"
        if with_target
        else "no_target"
    )

    output_dir = Path(
        "data/evaluation"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = output_dir / (
        f"{method}_{source_model}_"
        f"by_{evaluator}_{mode}.jsonl"
    )

    multi_path = None

    if method == "multi_agent":
        multi_dir = Path(
            "data/muti"
        )

        multi_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        multi_path = multi_dir / (
            f"multi_agent_roles_{source_model}_"
            f"by_{evaluator}_{mode}.jsonl"
        )

    done = load_done(
        output_path
    )

    pending = [
        generation
        for generation in generations
        if generation["sample_id"] not in done
    ]

    print(
        f"评估方法: {method}"
    )
    print(
        f"Evaluator: {evaluator}"
    )
    print(
        f"使用 target: {with_target}"
    )
    print(
        f"输入样本: {len(generations)}"
    )
    print(
        f"已完成: {len(done)}"
    )
    print(
        f"待评估: {len(pending)}"
    )
    print(
        f"输出: {output_path.resolve()}"
    )

    if method == "multi_agent":
        print(
            f"角色结果: {multi_path.resolve()}"
        )

    if not pending:
        print("没有待评估样本")
        return output_path

    for generation in pending:
        sample_id = generation["sample_id"]

        if sample_id not in cases_by_id:
            raise ValueError(
                f"cases 中找不到 sample_id: "
                f"{sample_id}"
            )

    with ModelService(
        evaluator
    ) as service:
        if service.base_url is None:
            raise RuntimeError(
                "模型服务没有 base_url"
            )

        with output_path.open(
            "a",
            encoding="utf-8",
        ) as f:
            total = len(pending)

            for index, generation in enumerate(
                pending,
                start=1,
            ):
                sample_id = (
                    generation["sample_id"]
                )

                print(
                    f"[{index}/{total}] "
                    f"{sample_id}"
                )

                result = evaluate_one(
                    case=cases_by_id[
                        sample_id
                    ],
                    generation=generation,
                    evaluator=evaluator,
                    base_url=service.base_url,
                    with_target=with_target,
                    method=method,
                )

                agent_results = result.pop(
                    "_agent_results"
                )

                if (
                    method == "multi_agent"
                    and multi_path is not None
                    and agent_results is not None
                ):
                    role_record = {
                        "sample_id": sample_id,
                        "source_backend": generation.get(
                            "backend"
                        ),
                        "source_model": generation.get(
                            "model"
                        ),
                        "evaluator_model": evaluator,
                        "with_target": with_target,
                        "agents": agent_results["agents"],
                        "adjudicator": agent_results["adjudicator"],
                    }

                    with multi_path.open(
                        "a",
                        encoding="utf-8",
                    ) as multi_file:
                        multi_file.write(
                            json.dumps(
                                role_record,
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

                        multi_file.flush()

                f.write(
                    json.dumps(
                        result,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                f.flush()

                print(
                    result["scores"]
                )

    print()
    print("评估完成")
    print(
        f"结果文件: "
        f"{output_path.resolve()}"
    )

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cases",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--evaluator",
        choices=MODELS,
        default=DEFAULT_EVALUATOR,
    )

    parser.add_argument(
        "--with-target",
        action="store_true",
    )

    parser.add_argument(
        "--limit",
        type=int,
    )
    parser.add_argument(
        "--method",
        choices=EVAL_METHODS,
        default="direct_llm",
    )

    args = parser.parse_args()

    evaluate_comments(
        cases_path=args.cases,
        input_path=args.input,
        evaluator=args.evaluator,
        with_target=args.with_target,
        limit=args.limit,
        method=args.method,
    )


if __name__ == "__main__":
    main()