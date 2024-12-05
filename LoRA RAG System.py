import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 设置模型和 tokenizer
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 准备模型
if torch.cuda.is_available():
    model = prepare_model_for_kbit_training(model)  

# 配置 LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["c_attn", "c_proj"],  
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# 构建简单的 RAG 系统
def retrieve_context(query, documents, top_k=1):
    """使用 TF-IDF 检索上下文"""
    vectorizer = TfidfVectorizer().fit(documents)
    doc_vectors = vectorizer.transform(documents)
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, doc_vectors).flatten()
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [documents[i] for i in top_indices], scores[top_indices]

# 文档集合
documents = [
    "Transformers are deep learning models for sequence-to-sequence tasks.",
    "Distilgpt2 is a lightweight language model designed for text generation.",
    "LoRA is a technique for parameter-efficient model fine-tuning.",
]

# 测试查询
query = "What is distilgpt2?"
retrieved_context, relevance_scores = retrieve_context(query, documents, top_k=3)

# 拼接 Prompt
prompt = f"Context: {retrieved_context[0]}\nQuery: {query}\nAnswer:"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(inputs.input_ids, max_length=50)
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Query: {query}\nRetrieved Context: {retrieved_context[0]}\nAnswer: {answer}")

# 模型评估
def evaluate_retrieval(queries, documents, ground_truths, top_k=3):
    """评估检索系统的 HitRate 和 MRR"""
    hit_count = 0
    reciprocal_ranks = []
    
    for query, ground_truth in zip(queries, ground_truths):
        retrieved_contexts, scores = retrieve_context(query, documents, top_k=top_k)
        hits = [ground_truth in context for context in retrieved_contexts]
        
        # 计算 HitRate
        hit_count += any(hits)
        
        # 计算 Reciprocal Rank
        if any(hits):
            rank = hits.index(True) + 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0)
    
    hit_rate = hit_count / len(queries)
    mrr = np.mean(reciprocal_ranks)
    return hit_rate, mrr

# 测试评估
test_queries = ["What are transformers?", "What is LoRA?", "What is distilgpt2?"]
ground_truths = [
    "Transformers are deep learning models for sequence-to-sequence tasks.",
    "LoRA is a technique for parameter-efficient model fine-tuning.",
    "Distilgpt2 is a lightweight language model designed for text generation.",
]
hit_rate, mrr = evaluate_retrieval(test_queries, documents, ground_truths, top_k=3)
print(f"HitRate: {hit_rate:.2f}, MRR: {mrr:.2f}")

#可以通过优化 向量检索（embedding） 部分来提高检索精度。例如，使用更高质量的嵌入模型（如 OpenAI Embedding 或更高级的模型）来增强检索阶段的表现。还可以调节 similarity_top_k 参数，以便检索更多的相关节点，提高检索模块的覆盖面。