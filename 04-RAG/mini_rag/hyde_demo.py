"""
HyDE（假设性文档嵌入）演示
==============================================
面试考点：HyDE 解决什么问题？和 Query Rewriting 的区别？
运行：uv run python "04-RAG/mini_rag/hyde_demo.py"
依赖：纯 numpy 模拟，无需 LLM/嵌入模型
"""

import numpy as np

print("=" * 60)
print("HyDE（Hypothetical Document Embeddings）演示")
print("=" * 60)

# ── 1. 问题：Query 和 Document 的语义鸿沟 ──────────────────
print("""
问题背景：
  用户查询通常是"问题形式"，而知识库文档是"陈述形式"。
  例如：
    Query:    "怎么减少大模型的幻觉？"
    Document: "RAG 通过检索外部知识来为模型提供事实依据，从而降低幻觉率。"

  两者语义相关，但表达形式差异大 → 向量相似度可能不高 → 检索效果差。
""")

# ── 2. HyDE 的核心思路 ──────────────────────────────────────
print("=" * 60)
print("HyDE 核心思路")
print("=" * 60)
print("""
传统检索：
  Query ──→ Embed ──→ q_vec ──→ 与 doc_vec 比较 ──→ 结果

HyDE 检索：
  Query ──→ LLM 生成假设性答案 ──→ Embed 假设答案 ──→ 与 doc_vec 比较 ──→ 结果

关键洞察：
  假设性答案虽然可能不准确，但它的"表达形式"更接近文档
  → 嵌入后的向量与真实文档更相似 → 检索效果更好
""")

# ── 3. 模拟演示 ──────────────────────────────────────────
print("=" * 60)
print("模拟演示")
print("=" * 60)

# 知识库文档
documents = [
    "RAG 通过检索外部知识库来为大模型提供事实依据，有效降低幻觉率。研究表明 RAG 可将幻觉率降低 30-50%。",
    "RLHF 通过人类反馈训练奖励模型，引导模型生成更符合人类偏好的回答，InstructGPT 将幻觉率从 41% 降至 21%。",
    "Self-Consistency 方法让模型多次采样生成答案，取多数投票结果，可以减少随机性导致的幻觉。",
    "Transformer 的自注意力机制通过 Q、K、V 矩阵计算注意力权重，捕获序列中的依赖关系。",
    "LoRA 通过低秩分解在冻结原始权重的基础上训练小矩阵，大幅减少微调所需的参数量和显存。",
]

query = "怎么减少大模型的幻觉？"

# 模拟 LLM 生成的假设性答案
hypothetical_answer = """减少大模型幻觉的方法包括：
1. 使用 RAG 检索增强生成，为模型提供外部知识作为事实依据
2. 通过 RLHF 人类反馈强化学习，训练模型生成更准确的回答
3. 采用 Self-Consistency 自一致性方法，多次采样取多数投票
4. 使用低温度采样减少随机性
5. 在 Prompt 中要求模型"如果不确定就说不知道"
"""

print(f"原始查询: \"{query}\"")
print(f"\nLLM 生成的假设性答案:")
print(f"  {hypothetical_answer[:100]}...")

# 模拟嵌入（用关键词重叠度模拟语义相似度）
def keyword_similarity(text1, text2):
    """用关键词重叠模拟语义相似度"""
    keywords = {"rag", "检索", "幻觉", "rlhf", "反馈", "奖励", "self-consistency",
                "一致性", "采样", "降低", "减少", "知识", "事实", "模型", "生成",
                "transformer", "注意力", "lora", "微调", "权重"}
    words1 = set(text1.lower().split()) | set(text1)
    words2 = set(text2.lower().split()) | set(text2)
    overlap = sum(1 for kw in keywords if (kw in text1.lower()) and (kw in text2.lower()))
    return overlap / len(keywords)

print(f"\n--- 相似度对比 ---")
print(f"{'文档':<55} {'原始Query':>8} {'HyDE':>8}")
print("-" * 75)

for doc in documents:
    sim_query = keyword_similarity(query, doc)
    sim_hyde = keyword_similarity(hypothetical_answer, doc)
    short = doc[:50] + "..."
    marker = " ✓" if sim_hyde > sim_query else ""
    print(f"{short:<55} {sim_query:>8.3f} {sim_hyde:>8.3f}{marker}")

print("""
→ HyDE 的假设性答案与相关文档的相似度更高
→ 因为假设答案的"表达形式"更接近文档（都是陈述句）
""")

# ── 4. HyDE vs Query Rewriting ──────────────────────────────
print("=" * 60)
print("HyDE vs Query Rewriting 对比")
print("=" * 60)
print("""
| 维度         | HyDE                        | Query Rewriting              |
|-------------|-----------------------------|-----------------------------|
| 做什么       | 生成假设性答案，用答案去检索     | 改写/扩展查询词，用新查询检索   |
| 解决什么问题  | Query-Doc 表达形式差异         | Query 表述不清/太短            |
| LLM 调用     | 1 次（生成假设答案）            | 1 次（改写查询）               |
| 嵌入对象     | 假设答案                      | 改写后的查询                   |
| 适用场景     | 问答式查询 vs 陈述式文档        | 用户表述模糊/口语化             |
| 风险         | 假设答案方向错误 → 检索偏差     | 改写丢失原始意图                |

最佳实践：两者可以结合使用
  Query → Rewrite → HyDE → 检索 → Rerank → 生成
""")

# ── 5. 面试要点 ──────────────────────────────────────────
print("=" * 60)
print("面试要点")
print("=" * 60)
print("""
1. HyDE 的核心洞察：假设答案不需要正确，只需要"形式上像文档"
2. HyDE 增加了一次 LLM 调用的延迟和成本
3. HyDE 对"事实性问题"效果好，对"观点性问题"可能引入偏差
4. HyDE 论文：Gao et al., 2022 "Precise Zero-Shot Dense Retrieval without Relevance Labels"
5. 实际生产中，HyDE 常与 Rerank 配合使用
""")
