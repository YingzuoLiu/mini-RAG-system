import numpy as np

# 假设
logits = np.array([2.5, 4.0, 1.8, 3.2, 2.9])
vocabulary = ["我", "喜欢", "学习", "编程", "人工智能"]

# 将logits转换为概率
def softmax(x):
    exp_x = np.exp(x)  
    return exp_x / exp_x.sum()

probabilities = softmax(logits)
print("词汇表:", vocabulary)
print("概率分布:", [round(p, 2) for p in probabilities])

# 贪婪搜索
greedy_choice = np.argmax(probabilities)
print("\n贪婪搜索选择:", vocabulary[greedy_choice])

# Top-k采样 (k=3)
k = 3
top_k_indices = np.argsort(probabilities)[-k:]  # 获取概率最高的k个词的索引
top_k_probs = probabilities[top_k_indices]      # 获取它们的概率

# 重新归一化
top_k_probs = top_k_probs / top_k_probs.sum()

# 从top-k中采样
sampled_index = np.random.choice(top_k_indices, p=top_k_probs)
print("Top-k采样选择:", vocabulary[sampled_index])
