# -*- coding: utf-8 -*-
"""
================================================================================
Mini RAG · 进阶版（Rerank 重排 + HyDE 假设文档检索）—— 纯 NumPy 原理版
================================================================================

【目的】
    在基线六步（main.py，不动）之上，加两招「进阶检索」，把 knowledge/RAG/高级篇/第8章
    讲的两个高频优化点变成可单步调试的代码：
        ① Rerank 重排   —— 检索后用 cross-encoder 对候选二次精排
        ② HyDE          —— 检索前把 query 改写成「假设答案」再去嵌入检索

    重点是"看清原理"，所以两招都用【纯规则/numpy】的 toy 实现演示机制；
    真实版（bge-reranker cross-encoder + ollama 生成假设答案）见 main_real.py。

【对应 knowledge/RAG 章节】
    第8章 8.1 查询转换（HyDE）        —— hyde_expand：假设性文档嵌入
    第8章 8.2 节点后处理器 / Rerank   —— rerank：cross-encoder 精排

【比喻】
    双塔检索（bi-encoder）= 海关「人脸识别闸机」：query 和 doc 各拍一张证件照
        （各自编码成向量），再比两张照片像不像 → 快，但只看"轮廓像不像"。
    Rerank（cross-encoder）= 海关「人工面谈」：把 query 和 doc 拉到一张桌前
        【面对面联合审问】，能问出证件照看不出的细节 → 慢，但准。
        所以工程套路是：闸机先放 top20 进来（快、广撒网），面谈再精排 top3（准）。
    HyDE = 你问"余弦相似度是啥"，文档里写的却是"余弦相似度衡量向量夹角"。
        query 与 doc 措辞不同 → 嵌入对不上。HyDE 让模型先【假装写一段答案】，
        用"答案腔"的文本去检索 → answer↔answer 比 query↔answer 更近。

【面试高频考点】
    1. 为什么要 Rerank？→ 双塔快但粗（各自编码+点积），cross-encoder 联合编码更准但慢
    2. 典型 pipeline？→ 向量召回较多候选(top20) → Rerank 精排 → 取 top3~5 送生成
    3. HyDE 为什么有效？→ 答案与答案在向量空间比 query 与答案更近（语义鸿沟）
    4. HyDE 的坑？→ 假设文档【只用于嵌入检索】，不替换原 query；rerank 仍用原始 query

【运行】
    cd /Users/sunhuanyu/ai-llm-learning
    uv run 04-RAG/mini_rag/main_advanced.py
    （依赖只有 numpy；复用同目录 main.py 的基线六步组件）

【注意 / 易错点】
    - Rerank 用【原始 query】打分，不是用 HyDE 改写后的文本！HyDE 只动「检索嵌入」。
    - toy 实现重在演示机制（cross-encoder 的"联合/覆盖"特征、HyDE 的"变换后检索"），
      效果是示意性的；真实效果需 cross-encoder 模型 / LLM（见 main_real.py）。
================================================================================
"""

from __future__ import annotations

import re
from typing import List, Tuple

import numpy as np

# 复用基线六步（main.py 保持不动，本文件只加"进阶检索"两招）
from main import (
    NumpyVectorStore,
    build_index,
    generate_dummy,
    retrieve,
)

# 候选条目类型：(chunk 文本, 分数, 元数据)
Candidate = Tuple[str, float, dict]

# 中文标点 / 空白 / 问号 —— 用于把句子切成"内容 bigram"前先清理
_CLEAN_RE = re.compile(r"[，。！？；：、“”‘’（）《》【】\s\?\!？！]+")


def _char_bigrams(text: str) -> List[str]:
    """取文本的字符 bigram（相邻两字），作为中文的轻量"短语单元"。

    main.py 的 embed 也基于 unigram+bigram，这里保持一致，便于对照。
    """
    cleaned = _CLEAN_RE.sub("", text)
    return [cleaned[i : i + 2] for i in range(len(cleaned) - 1)]


# ==============================================================================
# 第 7 步（进阶 A）：Rerank —— cross-encoder 对召回候选二次精排
#              对应 knowledge/RAG 第8章 8.2
# ==============================================================================
def cross_encoder_score(query: str, doc: str) -> float:
    """toy cross-encoder：query 的内容 bigram 在 doc 中的【覆盖率】∈ [0,1]。

    真实 cross-encoder（如 bge-reranker）把 [query, doc] 拼接喂 Transformer
    【联合编码】，能捕捉"query 这个说法在 doc 里到底成不成立"这类双向语义。
    这里用"query 的 bigram 有多少出现在 doc 里"来近似——它是个【联合特征】：
    必须同时拿到 query 和 doc 才能算。而双塔各自编码完只能算点积，拿不到这种
    "query 被 doc 覆盖了多少"的信号 → 这就是 Rerank 比纯向量排序更准的根本原因。
    """
    q_bis = _char_bigrams(query)
    if not q_bis:
        return 0.0
    doc_bi_set = set(_char_bigrams(doc))
    hits = sum(1 for b in q_bis if b in doc_bi_set)
    return hits / len(q_bis)


def rerank(query: str, candidates: List[Candidate], top_n: int = 3) -> List[Candidate]:
    """用 cross-encoder 对【双塔召回的候选】二次精排，取 top_n。

    输入 candidates：来自 retrieve 的双塔粗排结果，每条 (chunk, bi_score, meta)。
    输出：按 cross-encoder 分数降序的前 top_n 条；meta 里附 bi_score / rerank_score 便于对照。

    关键：打分用【原始 query】，不是 HyDE 文本（HyDE 只影响第②步的检索嵌入）。
    """
    scored: List[Candidate] = []
    for chunk, bi_score, meta in candidates:
        ce_score = cross_encoder_score(query, chunk)
        new_meta = dict(meta)
        new_meta["bi_score"] = bi_score       # 双塔分（粗排）
        new_meta["rerank_score"] = ce_score   # cross-encoder 分（精排）
        scored.append((chunk, ce_score, new_meta))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]


# ==============================================================================
# 第 7 步（进阶 B）：HyDE —— 检索前把 query 改写成「假设答案」
#              对应 knowledge/RAG 第8章 8.1
# ==============================================================================
# 中文提问骨架词（疑问/客套），HyDE 要把它们去掉，把"提问句"变成更像"文档/答案"的陈述
# 注意正则按"长串优先"排列，避免"什么是"被"是"抢先匹配
_QSKELETON = re.compile(
    r"(什么叫做|什么叫|什么是|为什么|为何|怎么样|怎么|如何|"
    r"到底是|到底|究竟|究竟能|"
    r"是用来干什么的|的作用是什么|有什么用|的作用|有什么|有何|是什么|"
    r"请问|一下|能不能|能否|可以|"
    r"呢|吗|啊|呀|吧|哦|么|\?|？|，|。)"
)


def hyde_expand(query: str) -> str:
    """无 LLM 的 toy HyDE：把 query 从「提问句」改写成「文档/答案腔」的陈述假设文本。

    真实 HyDE：让 LLM 针对 query 先写一段【假设答案】，再用这段假设答案去嵌入检索。
    核心洞察（knowledge/RAG 第8章 8.1）：用户提问与文档常有【语义鸿沟】，而
    answer↔answer 在向量空间比 query↔answer 更近 → 嵌入"假设答案"召回更好。

    没有 LLM 时，用规则做一次"轻量假设"：去掉提问骨架词（请问/什么是/到底/有什么/呢…）、
    保留内容词，得到一段更像文档的陈述文本。它能演示 HyDE 的【机制】（用一个变换后的
    文本去检索），并在"提问腔噪声重"的 query 上确实提升召回（见 demo）；
    真实语义级改写（LLM 生成假设答案）见 main_real.py。
    """
    stripped = _QSKELETON.sub("", query)
    stripped = re.sub(r"\s+", " ", stripped).strip()
    return stripped or query


# ==============================================================================
# 进阶检索编排：HyDE → 双塔多召回 → Rerank
# ==============================================================================
def retrieve_advanced(
    store: NumpyVectorStore,
    query: str,
    top_k: int = 3,
    recall_k: int = 8,
    use_hyde: bool = True,
    use_rerank: bool = True,
) -> List[Candidate]:
    """两阶段进阶检索：HyDE 改写检索文本 → 双塔多召回(recall_k) → cross-encoder 精排(top_k)。

    - use_hyde：检索用的【嵌入文本】改成 hyde_expand(query)；rerank 仍用原始 query。
    - use_rerank：对 recall_k 个双塔候选做 cross-encoder 精排。
    """
    # ① HyDE 只改【检索用】的文本
    retrieval_text = hyde_expand(query) if use_hyde else query
    if use_hyde and retrieval_text != query:
        print(f"   [HyDE] 原始 query : {query}")
        print(f"   [HyDE] 检索文本   : {retrieval_text}")

    # ② 双塔多召回（粗排）—— 故意召回 recall_k 个，给精排留选择空间
    candidates = retrieve(store, retrieval_text, top_k=recall_k)
    if not use_rerank:
        return candidates[:top_k]

    # ③ cross-encoder 精排（用【原始 query】打分！）
    return rerank(query, candidates, top_n=top_k)


# ==============================================================================
# 演示：把两招的效果"跑出来给人看"
# ==============================================================================
def _print_hits(title: str, hits: List[Candidate]) -> None:
    print(f"   {title}")
    if not hits:
        print("      (空)")
    for i, (c, s, m) in enumerate(hits):
        bi = m.get("bi_score")
        bi_str = f" 双塔={bi:.3f}" if bi is not None else ""
        ce = m.get("rerank_score")
        ce_str = f" 精排={ce:.3f}" if ce is not None else f" 分={s:.3f}"
        preview = c.replace("\n", " ")[:46]
        print(f"      [{i}] {ce_str}{bi_str} :: {preview}")


def demo_rerank(store: NumpyVectorStore) -> None:
    """演示①：双塔召回 top8 → cross-encoder 精排 top3，打印前/后顺序变化。"""
    query = "RAG 怎么生成答案"  # 相关：「第六步生成…生成答案」那段
    print("\n" + "=" * 70)
    print(f" 演示① Rerank —— query: {query}")
    print("=" * 70)

    # 双塔（bi-encoder）粗召回 8 个，标上 bi_score 方便对照
    candidates = retrieve(store, query, top_k=8)
    bi_tagged = [(c, s, {**m, "bi_score": s}) for c, s, m in candidates]
    _print_hits("双塔粗排 top8（各自编码+点积，快但粗）:", bi_tagged)

    # cross-encoder 精排 top3（用【原始 query】打分）
    refined = rerank(query, bi_tagged, top_n=3)
    _print_hits("Rerank 精排 top3（query+doc 联合覆盖打分）:", refined)

    print("   → 看：精排把【query 的关键短语 '生成答案' 真正覆盖到的】块从后面提进 top3；")
    print("        双塔与精排是【两套不同打分】，顺序因此不同。")
    print("   ⚠ 诚实说明：toy 的双塔与 cross-encoder 都基于'词重叠'，重排幅度有限；")
    print("     真实 cross-encoder（语义级联合编码）的精度提升见 main_real.py。")


def demo_hyde(store: NumpyVectorStore) -> None:
    """演示②：原始 query vs HyDE 改写后，召回的 top3 差异。"""
    query = "请问到底什么是余弦相似度呢"  # 提问腔噪声很重；相关：余弦相似度那段
    rewritten = hyde_expand(query)
    print("\n" + "=" * 70)
    print(f" 演示② HyDE —— query: {query}")
    print("=" * 70)
    print(f"   [HyDE] 原始 query : {query}")
    print(f"   [HyDE] 检索文本   : {rewritten}   （去掉 请问/到底/什么是/呢，只剩内容词）")

    raw_hits = retrieve(store, query, top_k=3)
    hyde_hits = retrieve(store, rewritten, top_k=3)
    _print_hits("直接用原始 query 检索 top3（提问腔稀释了内容词）:", raw_hits)
    _print_hits("用 HyDE 假设文档检索 top3（纯内容词，更聚焦）:", hyde_hits)

    # 直接量化"假设文档 vs 原始 query"哪个更接近【目标块】—— 把 HyDE 核心洞察亮出来
    target = "余弦相似度衡量两个向量方向上的接近程度"
    from main import embed

    sim_raw = float(embed(query) @ embed(target))
    sim_hyde = float(embed(rewritten) @ embed(target))
    print(f"   [量化] 原始 query 与目标块的余弦相似 = {sim_raw:.3f}")
    print(f"   [量化] HyDE 文本  与目标块的余弦相似 = {sim_hyde:.3f}")
    print("   → 看：假设文档（answer 腔）与目标块更接近 → 召回更准；")
    print("        真实 HyDE 让 LLM 写假设答案，能跨越更大的语义鸿沟（见 main_real.py）。")


def demo_full_pipeline(store: NumpyVectorStore) -> None:
    """演示③：HyDE + Rerank 完整进阶流水线，并接上生成。"""
    query = "检索回来的结果怎么排序"  # 相关：查询转换/重排序那段
    print("\n" + "=" * 70)
    print(f" 演示③ 完整进阶流水线（HyDE + Rerank）—— query: {query}")
    print("=" * 70)

    hits = retrieve_advanced(store, query, top_k=3, recall_k=8, use_hyde=True, use_rerank=True)
    _print_hits("进阶检索 top3:", hits)

    # 接基线的生成（防幻觉 + 抽取式回声）
    from main import build_prompt

    prompt = build_prompt(query, [c for c, _, _ in hits])
    answer = generate_dummy(prompt, hits)
    print("\n   [回答]")
    print("   " + answer.replace("\n", "\n   "))


def main() -> None:
    print("=" * 70)
    print(" Mini RAG · 进阶版（Rerank + HyDE）—— 纯 NumPy 原理版")
    print("=" * 70)

    # 复用基线建库（加载→分块→嵌入→入库）
    store = build_index(__import__("pathlib").Path(__file__).resolve().parent / "sample.txt",
                        chunk_sents=3, overlap=1)

    demo_rerank(store)
    demo_hyde(store)
    demo_full_pipeline(store)

    print("\n" + "=" * 70)
    print(" 进阶版跑通 ✓  → Rerank（cross-encoder 精排）+ HyDE（假设文档检索）")
    print(" 真实版（bge-reranker + ollama 生成的 HyDE）见 main_real.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
