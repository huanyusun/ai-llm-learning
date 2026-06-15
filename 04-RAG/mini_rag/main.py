# -*- coding: utf-8 -*-
"""
================================================================================
Mini RAG —— 纯 NumPy 自实现版（不依赖任何外部服务，直接 uv run 跑通）
================================================================================

【目的】
    用最少的依赖（仅 numpy，标准库）从零实现一条完整 RAG 流水线，
    把 knowledge/RAG/基础篇 里讲的"六步流水线"变成肉眼可见、可单步调试的代码。
    重点是"理解原理"，不是"生产可用"——真实版见本文件末尾的 REAL_VERSION_PLACEHOLDER 注释块。

【对应 knowledge/RAG 章节】
    第3章 3.1.1 原生代码开发 RAG        —— 整体六步流程
    第5章 数据加载与分割                 —— load_text / split_by_sentences / overlap
    第5章 5.3.2 SentenceSplitter         —— 分割优先级、chunk_size/overlap 是上限
    第6章 6.1 理解嵌入与向量             —— embed (这里用词袋哈希模拟)
    第6章 6.2 向量存储                   —— VectorStore (这里用 numpy 矩阵 + 余弦相似度)
    第7章 7.1 检索器                     —— retrieve top_k
    第7章 7.2 响应生成器                 —— build_prompt / generate (这里用抽取式回声模拟)

【为什么"嵌入"用哈希词袋而不是真模型？】
    纯 numpy 版的目标是"零外部依赖、直接跑通、看清原理"。
    真正的嵌入模型（sentence-transformers）要下几百 MB 权重、起 torch，
    与"理解 RAG 流水线骨架"这件事无关。这里用"字符/词 n-gram 哈希词袋"
    当一个 toy embedding：语义不强，但足以让"余弦相似度排序"这件事可见可验。
    想要真实语义向量，请切到真实版（见文件末）。

【运行】
    cd /Users/sunhuanyu/ai-llm-learning
    uv run 04-RAG/mini_rag/main.py
    （依赖只有 numpy，已在本仓库 pyproject.toml 的 dependencies 里）

【注意 / 易错点（连回 knowledge/RAG 第3章【易错/陷阱】）】
    - overlap 必须 0 <= overlap < chunk_size，否则相邻块退化为完全重复。
    - 构造索引时的 embed 函数必须与检索时同一个，否则向量空间不匹配、检索全错。
    - Chroma 默认 l2，文本嵌入一般用 cosine；本实现统一用 cosine。
================================================================================
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np

# 固定一个 toy embedding 的向量维度（哈希桶数量）。越大碰撞越少，但这里够用。
EMBED_DIM = 256


# ==============================================================================
# 第 1 步：加载与读取文档  —— 对应 第3章 3.1.1(1) / 第5章 数据加载
# ==============================================================================
def load_text(path: str | Path) -> str:
    """读取一个文本文件，返回纯文本内容。

    对应 knowledge/RAG 第3章 loadtext()：做路径检查 + 读取。
    这里简化为：存在性检查 + utf-8 解码。
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"文档不存在: {p}")
    return p.read_text(encoding="utf-8")


# ==============================================================================
# 第 2 步：分割文档（Chunking）—— 对应 第3章 3.1.1(2) / 第5章 5.3 SentenceSplitter
# ==============================================================================
def split_by_sentences(
    text: str,
    sentences_per_chunk: int = 3,
    overlap: int = 1,
) -> List[str]:
    """按中文句末标点分句，再每 N 句一个 Chunk，相邻 Chunk 重叠 overlap 句。

    参数约束（连回第3章面试考点 Q2 / 易错点 1）：
        sentences_per_chunk >= 1
        0 <= overlap < sentences_per_chunk

    overlap 的意义：让相邻块共享部分句子，避免在边界处丢失上下文，提升召回。
    """
    if sentences_per_chunk < 1:
        raise ValueError("sentences_per_chunk 必须 >= 1")
    if overlap < 0 or overlap >= sentences_per_chunk:
        raise ValueError("overlap 必须 0 <= overlap < sentences_per_chunk")

    # 用正则在中文句末标点后切句（连回第5章 5.3.2：split_by_regex 适合中文）
    sentences = re.split(r"(?<=[。！？\n])\s*", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return []

    chunks: List[str] = []
    step = sentences_per_chunk - overlap  # 步长 = 块大小 - 重叠
    if step <= 0:
        step = 1  # 防御：overlap 等于 chunk_size 的情况（已被上面拦截，双保险）

    i = 0
    while i < len(sentences):
        window = sentences[i : i + sentences_per_chunk]
        chunks.append("".join(window))
        i += step

    return chunks


# ==============================================================================
# 第 3 步：嵌入（Embedding）—— 对应 第6章 6.1 理解嵌入与向量
# ==============================================================================
# ---- 中文分词（toy 版）：按字符 + 标点切，再做 1/2-gram ----------------------
_PUNCT = re.compile(r"[，。！？；：、“”‘’（）《》【】\s]+")


def _tokenize(text: str) -> List[str]:
    """极简中文分词：去掉标点空白后，取单字 + 相邻二字 bigram 作为 token。

    真实场景会用 jieba/分词器/直接子词。这里够演示"相同词的文本向量更接近"。
    """
    cleaned = _PUNCT.sub("", text)
    unigrams = list(cleaned)
    bigrams = [cleaned[i : i + 2] for i in range(len(cleaned) - 1)]
    return unigrams + bigrams


def _hash_bucket(token: str) -> int:
    """把一个 token 哈希到 [0, EMBED_DIM) 的桶，做词袋特征的下标。"""
    h = hashlib.md5(token.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "little") % EMBED_DIM


def embed(text: str) -> np.ndarray:
    """toy embedding：词袋 + 哈希桶 + L2 归一化，返回单位向量。

    - 词袋（bag-of-words）思想：把文本表示成"每个 token 出现次数"的稀疏向量。
    - 哈希桶：用 hash 把无限词表压缩到固定维度（特征哈希 / hashing trick）。
    - L2 归一化：归一化后两向量点积 == 余弦相似度，省一次除法。

    局限：完全无序、无语义泛化，只是"词面重叠越多越相似"。
    它和真实 sentence-transformers 的关系：骨架一致（文本->定长向量->余弦比对），
    区别在真实模型用 Transformer 编码器学到语义，这里只用统计。
    """
    vec = np.zeros(EMBED_DIM, dtype=np.float32)
    for tok in _tokenize(text):
        vec[_hash_bucket(tok)] += 1.0
    # L2 归一化（零向量兜底，避免除 0）
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def embed_batch(texts: List[str]) -> np.ndarray:
    """批量嵌入，返回 [N, EMBED_DIM] 矩阵。"""
    return np.stack([embed(t) for t in texts]) if texts else np.zeros((0, EMBED_DIM), dtype=np.float32)


# ==============================================================================
# 第 4 步：向量存储（VectorStore）—— 对应 第6章 6.2 向量存储
# ==============================================================================
class NumpyVectorStore:
    """最朴素的向量存储：用 numpy 矩阵存所有 chunk 的向量，检索用暴力余弦。

    对应 knowledge/RAG 第6章 6.2.1 SimpleVectorStore（内存版）。
    生产里换成 Chroma/Milvus 等：底层是 HNSW 等近似最近邻，但接口一致（add/query）。
    """

    def __init__(self) -> None:
        self.texts: List[str] = []       # 每个 chunk 的原文
        self.metadatas: List[dict] = []  # 每个 chunk 的元数据（溯源用）
        self.matrix: np.ndarray = np.zeros((0, EMBED_DIM), dtype=np.float32)

    def add(self, texts: List[str], vectors: np.ndarray, metadatas: List[dict]) -> None:
        """加入一批 chunk 及其向量。"""
        if len(texts) != len(vectors) or len(texts) != len(metadatas):
            raise ValueError("texts / vectors / metadatas 长度不一致")
        self.texts.extend(texts)
        self.metadatas.extend(metadatas)
        self.matrix = np.vstack([self.matrix, vectors]) if len(self.texts) else vectors

    def query(self, query_vec: np.ndarray, top_k: int = 3) -> List[Tuple[str, float, dict]]:
        """用查询向量做余弦相似度检索，返回 top_k 个 (chunk, score, metadata)。

        因为存入和查询向量都已 L2 归一化，点积即余弦相似度（连回第6章 6.2.1）。
        """
        if len(self.texts) == 0:
            return []
        # 余弦相似度 = 归一化向量的点积
        scores = self.matrix @ query_vec  # shape [N]
        k = min(top_k, len(self.texts))
        # argpartition 拿 top_k 的下标，再按分数降序排（不全局排序，O(N) 而非 O(N logN)）
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [(self.texts[i], float(scores[i]), self.metadatas[i]) for i in top_idx]


# ==============================================================================
# 第 5 步：检索 top_k —— 对应 第7章 7.1 检索器
# ==============================================================================
def retrieve(store: NumpyVectorStore, query: str, top_k: int = 3):
    """把 query 嵌入成向量，去 store 里取 top_k。"""
    qv = embed(query)
    return store.query(qv, top_k=top_k)


# ==============================================================================
# 第 6 步：拼 Prompt + 生成 —— 对应 第7章 7.2 响应生成器 / 第3章 3.1.1(4)
# ==============================================================================
def build_prompt(query: str, contexts: List[str]) -> str:
    """把检索到的 chunk 与原问题组装成 Prompt（含防幻觉约束）。

    直接对应 knowledge/RAG 第3章 3.1.1(4) 的 modelquery 模板。
    """
    docs = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    return (
        "请基于以下上下文回答问题。如果上下文中不包含足够信息，"
        "请回答'我暂时无法回答该问题'，不要编造。\n\n"
        f"上下文：\n====\n{docs}\n====\n\n"
        f"我的问题是：{query}"
    )


def generate_dummy(prompt: str, contexts: List[Tuple[str, float, dict]]) -> str:
    """toy 生成器（无大模型依赖）。

    真实 RAG 的"生成"是调 LLM（Ollama/OpenAI）。
    这里没有大模型，就用"抽取 + 回声"模拟一个可运行的生成：
      - 取分数最高的一段上下文作为答案主体
      - 若最高分都很低，认为"未检索到" -> 触发防幻觉回复
    它存在的唯一目的是让整条流水线在零依赖下端到端跑完并输出有意义结果。
    """
    if not contexts:
        return "我暂时无法回答该问题（未检索到任何上下文）。"
    best_text, best_score, meta = contexts[0]
    # 经验阈值：toy embedding 的余弦分数若 < 0.15 视为基本无关（可调）
    if best_score < 0.15:
        return "我暂时无法回答该问题（上下文与问题相关度太低）。"
    source = meta.get("source", "未知来源")
    return f"[来自 {source} | 相似度 {best_score:.3f}]\n{best_text.strip()}"


# ==============================================================================
# 流水线编排：把上面六步串起来  —— 对应 第3章 3.1.1 整体
# ==============================================================================
def build_index(doc_path: str | Path, chunk_sents: int = 3, overlap: int = 1) -> NumpyVectorStore:
    """加载文档 -> 分块 -> 嵌入 -> 存入向量库。返回构造好的 store。"""
    print(f"[1/3] 加载文档: {doc_path}")
    text = load_text(doc_path)

    print("[2/3] 分块（SentenceSplitter 思路）...")
    chunks = split_by_sentences(text, sentences_per_chunk=chunk_sents, overlap=overlap)
    print(f"      得到 {len(chunks)} 个 chunk")
    for i, c in enumerate(chunks):
        preview = c.replace("\n", " ")[:40]
        print(f"        chunk[{i}]: {preview}...")

    print("[3/3] 嵌入 + 入库...")
    vectors = embed_batch(chunks)
    metas = [{"source": str(doc_path), "chunk_id": i} for i in range(len(chunks))]
    store = NumpyVectorStore()
    store.add(chunks, vectors, metas)
    print(f"      向量库已建立: {store.matrix.shape}")
    return store


def ask(store: NumpyVectorStore, query: str, top_k: int = 3) -> str:
    """对单条 query 跑 检索 -> 拼 prompt -> 生成。返回最终答案。"""
    print(f"\n>>> 问题: {query}")
    print("[检索] top_k =", top_k)
    hits = retrieve(store, query, top_k=top_k)
    for i, (c, s, m) in enumerate(hits):
        print(f"   hit[{i}] score={s:.3f} source={m['source']} :: {c.replace(chr(10), ' ')[:50]}")

    prompt = build_prompt(query, [c for c, _, _ in hits])
    print("[Prompt]\n" + prompt)
    answer = generate_dummy(prompt, hits)
    print("[回答]\n" + answer)
    return answer


# ==============================================================================
# main：默认跑一个内置 demo
# ==============================================================================
def main() -> None:
    here = Path(__file__).resolve().parent
    sample = here / "sample.txt"
    if not sample.exists():
        raise FileNotFoundError(f"示例知识文档不存在: {sample}")

    print("=" * 70)
    print(" Mini RAG —— 纯 NumPy 自实现版")
    print("=" * 70)

    store = build_index(sample, chunk_sents=3, overlap=1)

    # 几个示范问题（应能从 sample.txt 召回到正确段落）
    demo_questions = [
        "RAG 的全称是什么？",
        "overlap 重叠的作用是什么？",
        "向量检索一般用什么距离度量？",
        "今天晚饭吃什么？",  # 知识库里没有的问题，应触发防幻觉
    ]
    for q in demo_questions:
        ask(store, q, top_k=3)

    print("\n" + "=" * 70)
    print(" 全流程跑通 ✓")
    print(" 提示: 把 generate_dummy 换成真实的 Ollama/OpenAI 调用，")
    print("       把 embed 换成 sentence-transformers，就是一条真实 RAG。")
    print("       真实版见本目录 main_real.py（不自动安装依赖）。")
    print("=" * 70)


if __name__ == "__main__":
    main()


# ==============================================================================
#                              真实版说明（README 连接）
# ------------------------------------------------------------------------------
# 真实版（chromadb + sentence-transformers + ollama）写在另一个文件：
#     04-RAG/mini_rag/main_real.py
# 它与本文件一一对应同样的六步，只是把每个"toy 实现"换成真实组件：
#     load_text            -> 一样（标准库）
#     split_by_sentences   -> llama_index SentenceSplitter 或保留本实现
#     embed                -> sentence-transformers (bge-small-zh 等)
#     NumpyVectorStore     -> chromadb.PersistentClient + collection(cosine)
#     retrieve             -> collection.query(query_embeddings=..., n_results=k)
#     build_prompt         -> 一样（字符串模板）
#     generate_dummy       -> ollama.generate(model, prompt, stream=True)
# 依赖（需手动安装，不会自动跑）：
#     uv pip install chromadb sentence-transformers ollama
#     并本地起 ollama + pull 一个嵌入模型和一个生成模型。
# ==============================================================================
