from transformers import AutoTokenizer

# 以 Qwen2 为例，可以换成你实际用的模型
tokenizer = AutoTokenizer.from_pretrained("/volume/med-train/users/shuyan/lhg/VLM-R1/Qwen2.5-VL-3B-Instruct")

# 查看某个标点符号对应的 input_id
symbol = "。"
ids = tokenizer(symbol, add_special_tokens=False)["input_ids"]
print(f"符号 {symbol} 的 input_ids:", ids)

# 多个符号一起看
symbols = [".", ":", "!", ";"]
for s in symbols:
    print(s, "->", tokenizer(s, add_special_tokens=False)["input_ids"])