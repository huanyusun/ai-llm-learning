"""
Rerank 重排序演示 — 交叉编码器 vs 双塔编码器
==============================================
面试考点：为什么检索后还要 Rerank？交叉编码器和双塔的区别？
运行：uv run python "04-RAG/mini_rag/rerank_demo.py"
依赖：纯 numpy 实现，无需额外安装
"""

import numpy as np

print("=" * 60)
print("Rerank 重排序演示")
print("=" * 60)

# ── 1. 模拟初始检索结果 ──────────────────────────────────
query = "如何优化 RAG 系统的检索效果？"

# 模拟向量检索返回的 top-5 文档（按向量相似度排序）
retrieved_docs = [
    {"text": "RAG 系统由检索器和生成器两部分组成，检索器负责从知识库中找到相关文档。",
     "vec_score": 0.89},
    {"text": "向量数据库支持高效的近似最近邻搜索，常用算法包括 HNSW 和 IVF。",
     "vec_score": 0.85},
    {"text": "优化 RAG 检索效果的关键手段包括：混合检索、Rerank 重排序、查询改写和 HyDE。",
     "vec_score": 0.83},
    {"text": "Transformer 的自注意力机制计算复杂度为 O(n²)，限制了输入长度。",
     "vec_score": 0.78},
    {"text": "分块策略对检索质量影响很大，太大会引入噪声，太小会丢失上下文。",
     "vec_score": 0.76},
]

print(f"\n查询: \"{query}\"")
print(f"\n--- 初始检索结果（按向量相似度）---")
for i, doc in enumerate(retrieved_docs):
    print(f"  [{i+1}] (score={doc['vec_score']:.2f}) {doc['text'][:50]}...")

# ── 2. 模拟 Rerank（交叉编码器打分） ──────────────────────
print(f"\n--- Rerank 重排序 ---")
print("交叉编码器：将 query 和 doc 拼接后一起编码，计算相关性分数")
print("（真实场景用 cross-encoder 模型如 bge-reranker、cohere-rerank）\n")

# 模拟交叉编码器的相关性分数（真实场景中由模型计算）
# 关键：交叉编码器能捕捉 query-doc 之间的细粒度交互
rerank_scores = [0.72, 0.45, 0.95, 0.15, 0.68]

print(f"{'排名':<4} {'向量分':<8} {'Rerank分':<10} {'文档'}")
print("-" * 70)

# 按 rerank 分数重新排序
ranked_indices = np.argsort(rerank_scores)[::-1]
for new_rank, idx in enumerate(ranked_indices):
    doc = retrieved_docs[idx]
    marker = " ← 最相关！" if new_rank == 0 else ""
    print(f"  {new_rank+1}    {doc['vec_score']:.2f}     {rerank_scores[idx]:.2f}      "
          f"{doc['text'][:45]}...{marker}")

# ── 3. 双塔 vs 交叉编码器对比 ──────────────────────────────
print(f"\n" + "=" * 60)
print("双塔编码器 vs 交叉编码器")
print("=" * 60)
print("""
┌─────────────────────────────────────────────────────────┐
│  双塔编码器（Bi-Encoder）— 用于初始检索                    │
│                                                         │
│  Query ──→ [Encoder] ──→ q_vec ─┐                       │
│                                  ├→ cosine(q, d)        │
│  Doc   ──→ [Encoder] ──→ d_vec ─┘                       │
│                                                         │
│  ✅ 优点：doc 向量可预计算，检索速度快（毫秒级）            │
│  ❌ 缺点：query 和 doc 独立编码，无法捕捉细粒度交互        │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  交叉编码器（Cross-Encoder）— 用于 Rerank                 │
│                                                         │
│  [Query] [SEP] [Doc] ──→ [Encoder] ──→ relevance_score  │
│                                                         │
│  ✅ 优点：query 和 doc 联合编码，精度高                    │
│  ❌ 缺点：无法预计算，每对都要过一次模型，速度慢            │
└─────────────────────────────────────────────────────────┘

生产最佳实践：
  1. 初始检索（双塔）：从百万文档中快速召回 top-100
  2. Rerank（交叉编码器）：对 top-100 精排，选出 top-5
  3. 生成（LLM）：用 top-5 文档生成回答

常用 Rerank 模型：
  - bge-reranker-v2-m3（开源，多语言）
  - Cohere Rerank（API，效果好）
  - cross-encoder/ms-marco-MiniLM（英文）
""")

# ── 4. 多粒度分块对比 ──────────────────────────────────────
print("=" * 60)
print("附：多粒度分块对比实验")
print("=" * 60)

sample_text = """RAG（检索增强生成）是一种将信息检索与文本生成相结合的技术。
它的核心思想是：在生成回答之前，先从外部知识库中检索相关信息，
然后将检索到的信息作为上下文提供给大语言模型，从而生成更准确、更有依据的回答。
RAG 的主要优势包括：减少幻觉、知识可更新、可追溯来源。
RAG 的典型流程包括五个步骤：文档加载、文本分块、向量嵌入、相似度检索、生成回答。"""

chunk_sizes = [50, 100, 200]
print(f"\n原文长度: {len(sample_text)} 字符\n")

for size in chunk_sizes:
    chunks = [sample_text[i:i+size] for i in range(0, len(sample_text), size)]
    print(f"分块大小={size}: 产生 {len(chunks)} 个块")
    for j, chunk in enumerate(chunks):
        preview = chunk.replace('\n', ' ')[:60]
        print(f"  块{j}: \"{preview}...\" ({len(chunk)}字符)")
    print()

print("""分块策略面试要点：
- 太大：检索到的块含噪声多，LLM 上下文被浪费
- 太小：语义不完整，检索准确但信息不够
- 最佳实践：200-500 token，带 overlap（如 overlap=50）
- 高级：语义分块（按段落/主题切分）、递归分块（先大后小）
""")
