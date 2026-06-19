"""
混合检索演示 — BM25 + 向量检索 + RRF 融合
==============================================
面试考点：为什么需要混合检索？BM25 和向量检索各自的优缺点？
运行：uv run python "04-RAG/mini_rag/hybrid_search.py"
"""

import numpy as np
from collections import Counter
import math
import re

# ── 1. 准备文档 ──────────────────────────────────────────
documents = [
    "Transformer 使用自注意力机制来捕获序列中的长距离依赖关系。",
    "BERT 是一个基于 Transformer Encoder 的预训练语言模型，使用掩码语言模型任务。",
    "GPT 系列模型使用 Transformer Decoder 架构，通过自回归方式生成文本。",
    "RAG 将检索和生成结合，先从知识库检索相关文档，再用 LLM 生成回答。",
    "向量数据库如 Chroma、Milvus 用于存储和检索高维向量嵌入。",
    "BM25 是一种基于词频的经典检索算法，不需要深度学习模型。",
    "LoRA 通过低秩分解减少微调参数量，冻结原始权重只训练小矩阵。",
    "KV Cache 缓存历史的 Key 和 Value，避免重复计算，加速自回归推理。",
]

print("=" * 60)
print("混合检索演示：BM25 + 向量检索 + RRF 融合")
print("=" * 60)

# ── 2. BM25 检索 ──────────────────────────────────────────
print("\n--- BM25 检索 ---")

def tokenize(text):
    """简单中文分词（按字符 + 英文词）"""
    # 提取英文单词和中文字符
    tokens = re.findall(r'[a-zA-Z]+|[\u4e00-\u9fff]', text.lower())
    return tokens

class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.docs = docs
        self.doc_tokens = [tokenize(d) for d in docs]
        self.n = len(docs)
        self.avgdl = sum(len(t) for t in self.doc_tokens) / self.n
        # 计算 IDF
        self.idf = {}
        for tokens in self.doc_tokens:
            for t in set(tokens):
                self.idf[t] = self.idf.get(t, 0) + 1
        for t in self.idf:
            self.idf[t] = math.log((self.n - self.idf[t] + 0.5) / (self.idf[t] + 0.5) + 1)

    def score(self, query):
        q_tokens = tokenize(query)
        scores = []
        for doc_tokens in self.doc_tokens:
            s = 0
            tf_dict = Counter(doc_tokens)
            dl = len(doc_tokens)
            for qt in q_tokens:
                if qt in tf_dict:
                    tf = tf_dict[qt]
                    idf = self.idf.get(qt, 0)
                    s += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            scores.append(s)
        return np.array(scores)

bm25 = BM25(documents)

# ── 3. 向量检索（简化：用随机嵌入模拟） ──────────────────
print("--- 向量检索（模拟） ---")

np.random.seed(42)

def mock_embed(text):
    """模拟嵌入：基于关键词的简单向量化"""
    keywords = ["transformer", "attention", "bert", "gpt", "rag", "检索",
                "向量", "bm25", "lora", "微调", "cache", "生成"]
    tokens = tokenize(text)
    vec = np.zeros(len(keywords))
    for i, kw in enumerate(keywords):
        if kw in tokens or kw in text.lower():
            vec[i] = 1.0
    # 加点噪声让它更真实
    vec += np.random.randn(len(keywords)) * 0.1
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

doc_embeddings = [mock_embed(d) for d in documents]

def vector_search(query, top_k=8):
    q_emb = mock_embed(query)
    scores = [np.dot(q_emb, d_emb) for d_emb in doc_embeddings]
    return np.array(scores)

# ── 4. RRF（Reciprocal Rank Fusion）融合 ──────────────────
print("--- RRF 融合 ---\n")

def rrf_fusion(scores_list, k=60):
    """
    Reciprocal Rank Fusion
    对每个检索系统的排名取倒数求和：score = Σ 1/(k + rank_i)
    k=60 是常用默认值
    """
    n_docs = len(scores_list[0])
    fused = np.zeros(n_docs)
    for scores in scores_list:
        ranks = np.argsort(-scores)  # 降序排名
        for rank, doc_idx in enumerate(ranks):
            fused[doc_idx] += 1.0 / (k + rank + 1)
    return fused

# ── 5. 查询演示 ──────────────────────────────────────────
query = "Transformer 的注意力机制和检索"
print(f"查询: \"{query}\"\n")

bm25_scores = bm25.score(query)
vec_scores = vector_search(query)
rrf_scores = rrf_fusion([bm25_scores, vec_scores])

# 显示结果对比
print(f"{'文档':<45} {'BM25':>6} {'向量':>6} {'RRF':>8}")
print("-" * 70)
for i, doc in enumerate(documents):
    short = doc[:40] + "..." if len(doc) > 40 else doc
    print(f"{short:<45} {bm25_scores[i]:>6.2f} {vec_scores[i]:>6.3f} {rrf_scores[i]:>8.5f}")

# 排序结果
print(f"\n--- 最终排序（RRF）---")
ranked = np.argsort(-rrf_scores)
for rank, idx in enumerate(ranked[:3]):
    print(f"  Top-{rank+1}: [{idx}] {documents[idx][:50]}...")

# ── 6. 面试要点 ──────────────────────────────────────────
print("\n" + "=" * 60)
print("面试要点")
print("=" * 60)
print("""
1. BM25 优势：精确关键词匹配、无需 GPU、可解释性强
   BM25 劣势：不理解语义（"汽车"搜不到"轿车"）

2. 向量检索优势：语义理解、跨语言
   向量检索劣势：对专有名词/数字不敏感、需要嵌入模型

3. 混合检索 = BM25 + 向量检索，取长补短
   融合方法：RRF（简单有效）、加权求和、学习排序

4. RRF 公式：score(d) = Σ 1/(k + rank_i(d))
   - k=60 是经验值，防止排名靠前的文档权重过大
   - 不需要归一化分数，只用排名，跨系统可比

5. 生产实践：Elasticsearch 8.x 原生支持混合检索
""")
