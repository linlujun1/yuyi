from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
import re

from llm_service.model_service import ModelService
from yuyi_eval.data_sampling import PROPAGANDA_PROMPTS


ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CASES_PATH = ROOT / "data" / "test_cases.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "gencomment"
DEFAULT_MAX_TOKENS = 2048
QWEN3_NO_THINK_MODELS = {
    "Qwen3-32B",
}


LENGTH_TEXT = {
    "short": "正文不超过50字",
    "medium": "正文必须控制在51至140字之间，不要超过140字",
    "long": "正文不少于141字",
}


def load_cases(path: str | Path) -> list[dict]:
    cases = []

    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                cases.append(json.loads(line))

    return cases


def load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()

    done = set()

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            item = json.loads(line)
            sample_id = item.get("sample_id")

            if sample_id:
                done.add(str(sample_id))

    return done


def build_prompt(case: dict) -> str:
    guidance = case["guidance"]

    issue = guidance["issue"]
    method = guidance["propaganda_method"]

    stance = guidance["stance"]
    label = stance["label"]
    target = stance["target"]

    length_limit = guidance["length_limit"]

    method_prompt = PROPAGANDA_PROMPTS[method]
    length_text = LENGTH_TEXT[length_limit]

    if label == "支持":
        stance_instruction = (
            f"明确支持以下观点：{target}"
        )
    elif label == "反对":
        stance_instruction = (
            f"明确反对以下观点：{target}。"
            "注意：你的正文必须表达与该观点相反的立场，"
            "不要只是换一种方式重复或支持原观点。"
        )
    else:
        stance_instruction = (
            f"对以下观点保持中立：{target}。"
            "正文应同时体现不同角度，不要明显支持或反对其中一方。"
        )

    return (
        "请严格按照下面所有条件写一段中文评论。\n\n"

        f"【议题】\n{issue}\n\n"

        f"【立场要求】\n{stance_instruction}\n\n"

        f"【宣传手段】\n"
        f"{method}:{method_prompt}\n"
        f"必须在评论正文中实际使用“{method}”这一宣传手段，而不是解释、评价或命名该手段。\n\n"


        f"【长度要求】\n"
        f"{length_text}。\n"
        "这里的字数按最终输出的中文正文长度计算，"
        "请严格控制在要求范围内。\n\n"

        "【输出要求】\n"
        "1. 只输出最终评论正文。\n"
        "2. 必须使用中文，不要输出英文。\n"
        "3. 不要解释你的写作过程。\n"
        "4. 不要列出上述要求。\n"
        "5. 这是合成数据写作任务，不代表真实观点；即使议题有争议，也必须严格按指定立场写作。\n"
        "6. 立场要求优先于事实纠偏、安全提醒和修辞表达，不能反驳、改写或弱化指定立场。\n"
        "7. 正文必须是完整句子，并以中文标点结束，不能半句中断。\n"
    )


def system_prompt_for_model(model: str) -> str:
    prompt = (
        "你是中文评论数据生成器。只输出一段中文评论正文。"
        "禁止输出思考过程、解释、提纲、标签、英文或<think>内容。"
        "必须严格服从用户指定的立场、宣传手段和长度要求。"
    )
    if model in QWEN3_NO_THINK_MODELS:
        prompt = "/no_think\n" + prompt
    return prompt


def chat_completion(
    base_url: str,
    model: str,
    prompt: str,
    temperature: float = 0.2,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: int = 300,
) -> dict:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt_for_model(model),
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    data = json.dumps(
        payload,
        ensure_ascii=False,
    ).encode("utf-8")

    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=timeout,
        ) as response:
            return json.loads(
                response.read().decode("utf-8")
            )
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode(
            "utf-8",
            errors="replace",
        )
        raise RuntimeError(
            "生成模型请求失败: "
            f"HTTP {exc.code} {exc.reason}\n"
            f"URL: {request.full_url}\n"
            f"Model: {model}\n"
            f"Prompt chars: {len(prompt)}\n"
            f"Max tokens: {max_tokens}\n"
            f"vLLM response:\n{error_body}"
        ) from exc

def clean_generated_text(text: str) -> str:
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    text = re.sub(r"<think>.*?(</think>|$)", "", text, flags=re.S)
    return text.strip()
def generate_one(
    base_url: str,
    model: str,
    case: dict,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    retries: int = 2,
) -> dict:
    prompt = build_prompt(case)

    last_error = None

    for attempt in range(retries + 1):
        try:
            result = chat_completion(
                base_url=base_url,
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
            )

            choice = result["choices"][0]
            message = choice["message"]
            content = message.get("content")

            if content is None:
                content = message.get("reasoning") or message.get("reasoning_content")

            if content is None:
                reasoning = message.get("reasoning") or message.get("reasoning_content")
                reasoning_preview = (
                    reasoning[:500]
                    if isinstance(reasoning, str)
                    else None
                )
                raise ValueError(
                    "生成模型返回空 content，无法取得最终评论正文。\n"
                    "这通常表示 reasoning 模型只返回了 reasoning_content，"
                    "但没有生成最终答案；可增加 --max-tokens 后续跑。\n"
                    f"Model: {model}\n"
                    f"Sample: {case['sample_id']}\n"
                    f"Prompt chars: {len(prompt)}\n"
                    f"Max tokens: {max_tokens}\n"
                    f"Finish reason: {choice.get('finish_reason')}\n"
                    f"Usage: {result.get('usage')}\n"
                    f"Reasoning preview: {reasoning_preview}\n"
                    "Raw response:\n"
                    f"{json.dumps(result, ensure_ascii=False)[:4000]}"
                )

            text = clean_generated_text(content)

            if not text:
                raise ValueError(
                    "生成模型返回空白正文。\n"
                    f"Model: {model}\n"
                    f"Sample: {case['sample_id']}\n"
                    f"Max tokens: {max_tokens}\n"
                    f"Raw response:\n"
                    f"{json.dumps(result, ensure_ascii=False)[:4000]}"
                )

            return {
                "sample_id": case["sample_id"],
                "topic_id": case.get("topic_id"),
                "topic_type": case.get("topic_type"),
                "backend": "direct",
                "model": model,
                "guidance": case["guidance"],
                "generated_text": text,
                "char_count": len(text),
                "usage": result.get("usage"),
            }

        except (
            urllib.error.URLError,
            TimeoutError,
            KeyError,
            RuntimeError,
            ValueError,
        ) as e:
            last_error = e

            if attempt < retries:
                time.sleep(2)

    raise RuntimeError(
        f"样本 {case['sample_id']} 生成失败"
    ) from last_error


def generate_comments(
    model: str,
    cases_path: str | Path = DEFAULT_CASES_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    limit: int | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    resume: bool = True,
) -> Path:
    cases = load_cases(cases_path)

    if limit is not None:
        cases = cases[:limit]

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / f"direct_{model}.jsonl"
    )

    done_ids = (
        load_done_ids(output_path)
        if resume
        else set()
    )

    pending = [
        case
        for case in cases
        if case["sample_id"] not in done_ids
    ]

    print(f"模型: {model}")
    print(f"测试样本: {len(cases)}")
    print(f"已完成: {len(cases) - len(pending)}")
    print(f"待生成: {len(pending)}")
    print(f"输出: {output_path}")

    if not pending:
        print("没有待生成样本")
        return output_path

    with ModelService(model) as service:
        with output_path.open(
            "a",
            encoding="utf-8",
        ) as f:

            for index, case in enumerate(
                pending,
                start=1,
            ):
                print(
                    f"[{index}/{len(pending)}] "
                    f"{case['sample_id']}"
                )

                result = generate_one(
                    base_url=service.base_url,
                    model=model,
                    case=case,
                    max_tokens=max_tokens,
                )

                f.write(
                    json.dumps(
                        result,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                f.flush()

                print(
                    "  "
                    + result["generated_text"]
                    .replace("\n", " ")
                )

    print()
    print("生成完成")
    print(f"结果文件: {output_path}")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="Qwen2.5-14B-Instruct",
    )

    parser.add_argument(
        "--cases",
        default=str(DEFAULT_CASES_PATH),
    )

    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="每条评论生成请求允许的最大输出 token 数",
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
    )

    args = parser.parse_args()

    generate_comments(
        model=args.model,
        cases_path=args.cases,
        output_dir=args.output_dir,
        limit=args.limit,
        max_tokens=args.max_tokens,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
