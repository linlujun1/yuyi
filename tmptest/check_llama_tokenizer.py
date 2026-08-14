from transformers import AutoTokenizer, PreTrainedTokenizerFast

MODEL = "/model"

print("===== AutoTokenizer(use_fast=True) =====")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL,
    use_fast=True,
    trust_remote_code=True,
)

print("class =", type(tokenizer).__name__)
print("is_fast =", tokenizer.is_fast)
print("vocab_size =", len(tokenizer))

text = "你好，请严格按照要求写一段中文评论。"

ids = tokenizer.encode(text)

print("input =", repr(text))
print("ids =", ids)
print("count =", len(ids))
print("decode =", repr(tokenizer.decode(ids)))

print("\n===== Direct PreTrainedTokenizerFast =====")

fast_tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="/model/tokenizer.json",
)

ids2 = fast_tokenizer.encode(text)

print("class =", type(fast_tokenizer).__name__)
print("is_fast =", fast_tokenizer.is_fast)
print("vocab_size =", len(fast_tokenizer))
print("ids =", ids2)
print("count =", len(ids2))
print("decode =", repr(fast_tokenizer.decode(ids2)))