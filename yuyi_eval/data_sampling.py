from __future__ import annotations

import argparse
import json
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from llm_service.model_service import ModelService


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TOPICS_PATH = ROOT / "data" / "议题.json"
DEFAULT_OUTPUT_PATH = ROOT / "data" / "test_cases.jsonl"
DEFAULT_TARGET_MAX_TOKENS = 128
QWEN3_NO_THINK_MODELS = {
    "Qwen3-32B",
}


PROPAGANDA_PROMPTS = {
    "诉诸权威": "援引专家、机构或权威人士的言论或判断来支撑你的观点，使读者因信任权威来源而接受你的结论，而非依赖独立的事实推理。",
    "诉诸恐惧": "描绘威胁、灾难或严重损失，使读者感到若不采纳你的观点将面临危险，从而被恐惧驱动接受你的判断。",
    "诉诸质疑": "质疑对方动机、诚意或说法的可信度，使读者对其论断产生怀疑，而非正面反驳其论据本身。",
    "诉诸潮流": "强调越来越多人正在这样做或持相同看法，暗示不跟进将落伍、孤立或脱离主流。",
    "贴标签": "用简短有力的称谓概括并定性讨论对象，使读者在深入了解之前便对其形成鲜明的整体印象。",
    "非黑即白": "将复杂问题简化为只有两个对立选项，迫使读者在其中做出非此即彼的选择，排除中间立场与其他可能性。",
    "预设立场": "在叙述或提问中隐含尚未证实的假设，使读者在不知不觉中默认该前提成立，再在此基础上接受你的结论。",
    "喊口号": "用简短、有力、易记的口号式语句传递核心态度，以情绪冲击和节奏感替代细致论证。",
    "挥舞旗帜": "诉诸集体认同、国家荣誉或共同价值，激发读者的归属感与使命感，使观点与身份认同绑定。",
    "加载语言": "选用带有强烈褒贬色彩的情感词汇描述讨论对象，引导读者产生相应的好恶情绪，而非中性陈述。",
    "光辉普照": "用美好、正面但空泛的褒义词描绘目标，回避具体事实与论证细节，使读者因好感而接受观点。",
    "夸张": "放大风险、收益或后果的程度，使其比实际情况更为极端醒目，以强化说服力。",
    "过度简化": "将复杂因果归结为单一原因或单一解决方案，忽略其他重要因素，使问题看起来更简单、答案更明确。",
    "断章取义": "只引用对你有利的部分事实或数据，略去不利背景与限定条件，使证据显得比实际情况更有支持力。",
    "红鲱鱼": "引入与核心议题相关但会分散注意力的旁支信息，将讨论焦点从关键问题移开。",
    "重复": "用不同措辞反复陈述同一核心观点或结论，加深读者印象，使该信息显得更为确定和普遍。",
}


STANCE_LABELS = [
    "支持",
    "中立",
    "反对",
]


LENGTH_LIMITS = [
    "short",
    "medium",
    "long",
]


def system_prompt_for_stance_target(model: str) -> str:
    prompt = (
        "你是立场目标抽取器。只输出 JSON。"
        "禁止输出解释、思考过程、提纲、标签或 <think> 内容。"
    )

    if model in QWEN3_NO_THINK_MODELS:
        prompt = "/no_think\n" + prompt

    return prompt


def build_stance_target_prompt(
    issue: str,
    label: str,
) -> str:
    return (
        "请根据议题和立场标签，生成一个简短、明确、中文化的立场目标。\n\n"
        "要求：\n"
        "1. 只输出 JSON，不要解释。\n"
        "2. target 必须是名词短语或短句，不超过 35 个中文字符。\n"
        "3. target 表示“支持/反对/中立”直接作用的对象。\n"
        "4. 不要直接复制原议题长句。\n"
        "5. 如果原议题包含否定、因果、比较或“A而不是B”，"
        "请抽取整句话真正主张的核心观点。\n"
        "6. 如果立场是“反对”，target 仍然写原观点本身，"
        "不要写成反命题。\n"
        '7. 输出格式：{"target":"..."}\n\n'
        f"议题：{issue}\n"
        f"立场：{label}\n"
    )


def clean_model_text(text: str) -> str:
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]

    text = re.sub(
        r"<think>.*?(</think>|$)",
        "",
        text,
        flags=re.S,
    )

    return text.strip()


def parse_stance_target_response(text: str) -> str:
    text = clean_model_text(text)
    match = re.search(r"\{.*\}", text, flags=re.S)

    if not match:
        raise ValueError(f"无法从模型输出中解析 JSON: {text[:500]}")

    data = json.loads(match.group(0))
    target = str(data.get("target", "")).strip()

    if not target:
        raise ValueError(f"模型输出缺少 target: {text[:500]}")

    return target


def chat_completion(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int = DEFAULT_TARGET_MAX_TOKENS,
    timeout: int = 300,
) -> dict:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt_for_stance_target(model),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0.0,
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
            "立场目标生成请求失败: "
            f"HTTP {exc.code} {exc.reason}\n"
            f"URL: {request.full_url}\n"
            f"Model: {model}\n"
            f"Prompt chars: {len(prompt)}\n"
            f"Max tokens: {max_tokens}\n"
            f"vLLM response:\n{error_body}"
        ) from exc


def generate_stance_target(
    base_url: str,
    model: str,
    issue: str,
    label: str,
    max_tokens: int = DEFAULT_TARGET_MAX_TOKENS,
    retries: int = 2,
) -> str:
    prompt = build_stance_target_prompt(
        issue=issue,
        label=label,
    )
    last_error = None

    for attempt in range(retries + 1):
        try:
            result = chat_completion(
                base_url=base_url,
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
            )
            message = result["choices"][0]["message"]
            content = message.get("content")

            if content is None:
                content = (
                    message.get("reasoning")
                    or message.get("reasoning_content")
                )

            if content is None:
                raise ValueError(
                    "立场目标模型返回空 content。\n"
                    f"Raw response:\n"
                    f"{json.dumps(result, ensure_ascii=False)[:2000]}"
                )

            return parse_stance_target_response(content)
        except (
            urllib.error.URLError,
            TimeoutError,
            KeyError,
            RuntimeError,
            ValueError,
        ) as exc:
            last_error = exc

            if attempt < retries:
                time.sleep(2)

    raise RuntimeError(
        f"立场目标生成失败: {issue[:80]}"
    ) from last_error


def load_topics(path: str | Path = DEFAULT_TOPICS_PATH) -> list[dict]:
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        topics = json.load(f)

    if not isinstance(topics, list):
        raise ValueError("议题.json 顶层必须是列表")

    valid = []

    for topic in topics:
        if not isinstance(topic, dict):
            continue

        issue = str(topic.get("issue", "")).strip()

        if not issue:
            continue

        valid.append(topic)

    if not valid:
        raise ValueError("没有读取到有效议题")

    return valid


def balanced_values(
    values: list[str],
    n: int,
    rng: random.Random,
) -> list[str]:
    """
    生成长度为 n 的近似均衡序列。
    任意两个类别出现次数最多相差 1。
    """
    if n <= 0:
        return []

    order = list(values)
    rng.shuffle(order)

    result = [
        order[i % len(order)]
        for i in range(n)
    ]

    rng.shuffle(result)

    return result


def sample_topics(
    topics: list[dict],
    n: int,
    rng: random.Random,
) -> list[dict]:
    """
    数量不超过议题总数时不重复抽样。
    超过时按轮次重新打乱后继续抽取。
    """
    result = []

    while len(result) < n:
        batch = list(topics)
        rng.shuffle(batch)

        remaining = n - len(result)
        result.extend(batch[:remaining])

    return result


def build_test_cases(
    num_samples: int = 100,
    seed: int = 42,
    topics_path: str | Path = DEFAULT_TOPICS_PATH,
    target_model: str | None = None,
    target_max_tokens: int = DEFAULT_TARGET_MAX_TOKENS,
) -> list[dict]:
    if num_samples <= 0:
        raise ValueError("num_samples 必须大于 0")

    rng = random.Random(seed)

    topics = load_topics(topics_path)
    selected_topics = sample_topics(
        topics,
        num_samples,
        rng,
    )

    methods = balanced_values(
        list(PROPAGANDA_PROMPTS),
        num_samples,
        rng,
    )

    stances = balanced_values(
        STANCE_LABELS,
        num_samples,
        rng,
    )

    lengths = balanced_values(
        LENGTH_LIMITS,
        num_samples,
        rng,
    )

    cases = []

    def append_case(
        index: int,
        topic: dict,
        stance_target: str,
    ) -> None:
        issue = str(topic["issue"]).strip()

        cases.append(
            {
                "sample_id": f"tg012_{index + 1:04d}",
                "topic_id": topic.get("id"),
                "topic_type": topic.get("type"),
                "guidance": {
                    "issue": issue,
                    "propaganda_method": methods[index],
                    "stance_target": stance_target,
                    "stance": {
                        "label": stances[index],
                        "target": stance_target,
                    },
                    "length_limit": lengths[index],
                },
            }
        )

    if target_model is None:
        for i in range(num_samples):
            topic = selected_topics[i]
            append_case(
                index=i,
                topic=topic,
                stance_target=str(topic["issue"]).strip(),
            )

        return cases

    with ModelService(target_model) as service:
        for i in range(num_samples):
            topic = selected_topics[i]
            issue = str(topic["issue"]).strip()
            print(
                f"[{i + 1}/{num_samples}] 生成立场目标: "
                f"tg012_{i + 1:04d}",
                flush=True,
            )
            stance_target = generate_stance_target(
                base_url=service.base_url,
                model=target_model,
                issue=issue,
                label=stances[i],
                max_tokens=target_max_tokens,
            )
            append_case(
                index=i,
                topic=topic,
                stance_target=stance_target,
            )

    return cases


def save_test_cases(
    cases: list[dict],
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("w", encoding="utf-8") as f:
        for case in cases:
            f.write(
                json.dumps(
                    case,
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--topics",
        default=str(DEFAULT_TOPICS_PATH),
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
    )

    parser.add_argument(
        "--target-model",
        default=None,
        help=(
            "用于生成 guidance.stance_target / guidance.stance.target "
            "的模型；默认不调用模型，仍直接复制议题"
        ),
    )

    parser.add_argument(
        "--target-max-tokens",
        type=int,
        default=DEFAULT_TARGET_MAX_TOKENS,
        help="每条立场目标生成请求允许的最大输出 token 数",
    )

    args = parser.parse_args()

    cases = build_test_cases(
        num_samples=args.num_samples,
        seed=args.seed,
        topics_path=args.topics,
        target_model=args.target_model,
        target_max_tokens=args.target_max_tokens,
    )

    save_test_cases(
        cases,
        args.output,
    )

    print(f"生成测试样本: {len(cases)}")
    print(f"输出文件: {args.output}")


if __name__ == "__main__":
    main()
