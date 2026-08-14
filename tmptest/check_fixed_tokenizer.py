from transformers import AutoTokenizer

TOKENIZER_PATH = (
    "/user_home/linlujun/linlujun/model/"
    "DeepSeek-R1-Distill-Llama-8B-tokenizer-fixed"
)

TEXT = "你好，请严格按照要求写一段中文评论。"


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH,
        use_fast=True,
        trust_remote_code=True,
    )

    print("===== tokenizer info =====")
    print("class =", type(tokenizer).__name__)
    print("is_fast =", tokenizer.is_fast)
    print("vocab_size =", len(tokenizer))

    ids = tokenizer.encode(TEXT)

    print()
    print("===== encode / decode =====")
    print("input =", repr(TEXT))
    print("ids =", ids)
    print("count =", len(ids))
    print("decode =", repr(tokenizer.decode(ids)))

    messages = [
        {
            "role": "user",
            "content": TEXT,
        }
    ]

    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    print()
    print("===== chat template =====")
    print(repr(rendered))


if __name__ == "__main__":
    main()