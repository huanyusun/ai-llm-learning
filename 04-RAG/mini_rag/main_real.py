# -*- coding: utf-8 -*-
"""
================================================================================
Mini RAG —— 真实版（chromadb + sentence-transformers + ollama）
================================================================================

【依赖（不会自动安装，请手动装）】
    uv pip install chromadb sentence-transformers ollama

    另外需要本地起一个 Ollama 服务，并 pull 模型：
        ollama pull bge-m3            # 嵌入模型（也可用 nomic-embed-text 等）
        ollama pull qwen2.5:7b        # 生成模型（任意 instruct 模型均可）
        ollama serve                  # 默认监听 http://localhost:11434

    说明：本文件用 ollama 的 embeddings 接口做嵌入（与生成模型同源、零额外配置）。
    如果你更想用 sentence-transformers 本地跑嵌入，把 embed() 换成：
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
        def embed(text): return _model.encode(text, normalize_embeddings=True)
    其余六步完全不变（这正是把"原理"与"组件"解耦的意义）。

【与 main.py（纯 numpy 版）的对应关系】
    六步一一对应，只是把每个 toy 实现换成真实组件：

    | 步骤  | main.py (toy)             | main_real.py (真实)                       |
    |-------|---------------------------|-------------------------------------------|
    | 加载  | load_text (标准库)         | load_text (标准库，不变)                   |
    | 分块  | split_by_sentences (正则)  | split_by_sentences (同实现；也可换 SentenceSplitter) |
    | 嵌入  | embed (哈希词袋)           | ollama.embeddings(model=bge-m3)           |
    | 存储  | NumpyVectorStore (numpy)   | chromadb PersistentClient + collection(cosine) |
    | 检索  | store.query(余弦)          | collection.query(n_results=k)             |
    | 生成  | generate_dummy (抽取)      | ollama.generate(stream=True) 真实大模型    |

【对应 knowledge/RAG 章节】
    第2章 开发环境搭建             —— Ollama / Chroma / 嵌入模型
    第3章 3.1.1 原生代码 RAG        —— 整体六步
    第5章 数据加载与分割            —— split_by_sentences
    第6章 6.2.2 Chroma 向量存储     —— PersistentClient / hnsw:space=cosine
    第7章 7.1/7.2 检索与生成        —— collection.query / ollama.generate
    （想用 LlamaIndex 一行搞定，见第3章 3.1.2 的 5 行代码 RAG）

【运行】
    cd /Users/sunhuanyu/ai-llm-learning
    uv run 04-RAG/mini_rag/main_real.py
    （先确保上面的依赖装好、ollama 起好、模型 pull 好）
================================================================================
"""

from __future__ import annotations

# ---- 依赖（需要手动安装，不会自动 pip install）------------------------------
# import chromadb            # pip install chromadb
# import ollama              # pip install ollama
# -----------------------------------------------------------------------------

import re
import sys
from pathlib import Path
from typing import List, Tuple

# --- 模型配置（按你本地 ollama pull 的模型改）---------------------------------
EMBED_MODEL = "bge-m3"       # 嵌入模型
LLM_MODEL = "qwen2.5:7b"     # 生成模型
CHROMA_DIR = ".chroma_db"    # Chroma 持久化目录（相对本文件）
COLLECTION = "mini_rag"


# ==============================================================================
# 第 1 步：加载文档（与纯 numpy 版完全相同）
# ==============================================================================
def load_text(path: str | Path) -> str:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"文档不存在: {p}")
    return p.read_text(encoding="utf-8")


# ==============================================================================
# 第 2 步：分块（与纯 numpy 版相同的 SentenceSplitter 思路；也可换 LlamaIndex）
# ==============================================================================
def split_by_sentences(text: str, sentences_per_chunk: int = 3, overlap: int = 1) -> List[str]:
    if sentences_per_chunk < 1:
        raise ValueError("sentences_per_chunk 必须 >= 1")
    if overlap < 0 or overlap >= sentences_per_chunk:
        raise ValueError("overlap 必须 0 <= overlap < sentences_per_chunk")
    sentences = re.split(r"(?<=[。！？\n])\s*", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return []
    chunks: List[str] = []
    step = max(1, sentences_per_chunk - overlap)
    i = 0
    while i < len(sentences):
        chunks.append("".join(sentences[i : i + sentences_per_chunk]))
        i += step
    return chunks


# ==============================================================================
# 第 3 步：嵌入（真实嵌入模型，经 Ollama）
# ==============================================================================
def embed(text: str) -> List[float]:
    """调用 Ollama 嵌入接口，返回一个 float 向量。

    对应 knowledge/RAG 第3章：ollama.embeddings(model=embedmodel, prompt=chunk)['embedding']
    """
    import ollama  # 延迟导入，避免没装依赖时 import 阶段就崩

    return ollama.embeddings(model=EMBED_MODEL, prompt=text)["embedding"]


def embed_batch(texts: List[str]) -> List[List[float]]:
    # ollama.embeddings 是单条接口；批量就循环。大库可换 ollama 新版 embed() 批量接口。
    return [embed(t) for t in texts]


# ==============================================================================
# 第 4 步：向量存储（Chroma）—— 对应 第6章 6.2.2
# ==============================================================================
def get_collection():
    """创建/打开一个 Chroma collection（持久化到磁盘）。

    关键点（连回第3章【易错/陷阱】7、8）：
      - hnsw:space=cosine：文本嵌入默认推荐余弦，不指定会用默认 l2 导致排序失真。
      - 复用 collection 名前应清空旧数据，避免与新数据混叠。
    """
    import chromadb

    here = Path(__file__).resolve().parent
    client = chromadb.PersistentClient(path=str(here / CHROMA_DIR))
    # 清掉同名旧集合，保证可重复运行（生产里别这么干，用 upsert）
    try:
        client.delete_collection(name=COLLECTION)
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


# ==============================================================================
# 流水线：建索引
# ==============================================================================
def build_index(doc_path: str | Path, chunk_sents: int = 3, overlap: int = 1):
    print(f"[1/3] 加载文档: {doc_path}")
    text = load_text(doc_path)

    print("[2/3] 分块...")
    chunks = split_by_sentences(text, sentences_per_chunk=chunk_sents, overlap=overlap)
    print(f"      得到 {len(chunks)} 个 chunk")

    print("[3/3] 嵌入(ollama) + 入库(chroma)...")
    vectors = embed_batch(chunks)
    collection = get_collection()
    collection.add(
        ids=[f"chunk-{i}" for i in range(len(chunks))],
        embeddings=vectors,
        documents=chunks,
        metadatas=[{"source": str(doc_path), "chunk_id": i} for i in range(len(chunks))],
    )
    print(f"      入库 {len(chunks)} 条")
    return collection


# ==============================================================================
# 第 5/6 步：检索 + 拼 prompt + 生成
# ==============================================================================
def retrieve(collection, query: str, top_k: int = 3):
    """对应 第3章 collection.query(query_embeddings=..., n_results=k)。"""
    qv = embed(query)
    res = collection.query(query_embeddings=[qv], n_results=top_k)
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res.get("distances", [[0.0] * len(docs)])[0]
    return list(zip(docs, dists, metas))


def build_prompt(query: str, contexts: List[str]) -> str:
    docs = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    return (
        "请基于以下上下文回答问题。如果上下文中不包含足够信息，"
        "请回答'我暂时无法回答该问题'，不要编造。\n\n"
        f"上下文：\n====\n{docs}\n====\n\n"
        f"我的问题是：{query}"
    )


def generate(prompt: str) -> str:
    """调 Ollama 生成模型，流式输出。对应 第3章 ollama.generate(stream=True)。"""
    import ollama

    out = []
    stream = ollama.generate(model=LLM_MODEL, prompt=prompt, stream=True)
    for chunk in stream:
        piece = chunk.get("response", "")
        if piece:
            print(piece, end="", flush=True)  # flush=True：连回易错点 9
            out.append(piece)
    print()
    return "".join(out)


def ask(collection, query: str, top_k: int = 3) -> str:
    print(f"\n>>> 问题: {query}")
    hits = retrieve(collection, query, top_k=top_k)
    for i, (c, d, m) in enumerate(hits):
        print(f"   hit[{i}] dist={d:.4f} source={m['source']} :: {c.replace(chr(10), ' ')[:50]}")
    prompt = build_prompt(query, [c for c, _, _ in hits])
    print("[生成回答]")
    return generate(prompt)


def main() -> None:
    # 延迟导入，给出友好提示而不是裸 ImportError
    try:
        import chromadb  # noqa: F401
        import ollama  # noqa: F401
    except ImportError as e:
        print("缺少依赖。请先手动安装：", file=sys.stderr)
        print("  uv pip install chromadb sentence-transformers ollama", file=sys.stderr)
        print(f"原始错误: {e}", file=sys.stderr)
        sys.exit(1)

    here = Path(__file__).resolve().parent
    sample = here / "sample.txt"
    print("=" * 70)
    print(" Mini RAG —— 真实版 (chromadb + ollama)")
    print(f" embed={EMBED_MODEL}  llm={LLM_MODEL}")
    print("=" * 70)

    collection = build_index(sample, chunk_sents=3, overlap=1)

    for q in [
        "RAG 的全称是什么？",
        "overlap 重叠的作用是什么？",
        "向量检索一般用什么距离度量？",
    ]:
        ask(collection, q, top_k=3)


if __name__ == "__main__":
    main()
