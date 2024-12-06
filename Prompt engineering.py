# 构建 Prompt：包含上下文、问题、以及逐步推理的示例
def build_prompt(query, context, cot=False):
    """
    构建 Prompt 的函数，支持 CoT 模式
    """
    if cot:
        # 使用 Chain-of-Thought 模式，提供示例
        prompt = (
            f"Context: {context}\n"
            f"Query: {query}\n"
            "Answer step by step:\n"
            "1. First, identify key information in the context.\n"
            "2. Use the key information to answer the query.\n"
            "Answer:"
        )
    else:
        # 简单直接的 Prompt
        prompt = f"Context: {context}\nQuery: {query}\nAnswer:"
    
    return prompt

# 示例输入
query = "What is LoRA?"
retrieved_context = "LoRA is a technique for parameter-efficient model fine-tuning."
prompt = build_prompt(query, retrieved_context, cot=True)

# 模型输入与生成
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(inputs.input_ids, max_length=100)

# 打印结果
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
