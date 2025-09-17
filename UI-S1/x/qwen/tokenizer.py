qwen_tokenizer = None

def get_text_seq_len(text):
    global qwen_tokenizer

    if qwen_tokenizer is None:
        from transformers import AutoTokenizer
        qwen_tokenizer = AutoTokenizer.from_pretrained("checkpoints/Qwen/Qwen2-VL-7B-Instruct")
    return len(qwen_tokenizer(text)['input_ids'])