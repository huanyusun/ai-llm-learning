# -*- coding: utf-8 -*-
"""
================================================================================
Mini RAG · 评估器（从零，零依赖）—— faithfulness / relevancy / context_relevance
================================================================================

【目的】
    RAG 系统做完不算完——还得能【量化它答得好不好】。本文件从零实现 RAG 评估的三个核心
    指标，直接给我们前面做的 mini_rag 打分，把 knowledge/RAG 第10章「评估」变成可跑的代码。
    （全景图里「评估」此前是整块空白，本文件补上。）

【三个指标（面试要能解释）】
    1. faithfulness 忠实度：答案是否被【检索到的上下文】支撑 → 防幻觉。
       答案凭空编的内容，上下文里没有 → 忠实度低。
    2. relevancy 相关度：答案是否【切题】回答了问题 → 防答非所问。
    3. context_relevance 检索相关性：召回的上下文是否与问题相关 → 评估检索质量。
    这里的实现是【规则版】（字符 bigram 覆盖率）当 toy proxy；真实版用 LLM-as-judge
    逐句判定（ragas 的 faithfulness/relevancy），见文末附录。

【对应 knowledge/RAG】第10章 评估 RAG 应用（faithfulness 忠实度 / relevancy 相关度）。

【运行】
    cd /Users/sunhuanyu/ai-llm-learning
    uv run 04-RAG/mini_rag/eval_rag.py
    （复用同目录 main.py 的基线 RAG；依赖只有 numpy）

【注意】
    - 规则版指标是 toy proxy（靠词面重叠近似语义），重在演示【指标定义与评估流程】，
      精确语义判定要靠 LLM-as-judge（附录 ragas）。
    - 对【超纲/无据】问题，context_relevance 低 + RAG 拒答才是正确行为（不是 faithfulness 低）。
================================================================================
"""

from __future__ import annotations

import re
from pathlib import Path

from main import build_index, build_prompt, generate_dummy, retrieve

# 候选条目类型
Case = tuple  # (query, kind)  kind ∈ {"in_scope", "out_of_scope"}

# 字符 bigram（中文友好）：用于规则版指标
_CLEAN = re.compile(r"[，。！？；：、“”‘’（）《》【】\s\?\!？！]+")


def _bigrams(text: str) -> set[str]:
    cleaned = _CLEAN.sub("", text)
    return {cleaned[i : i + 2] for i in range(len(cleaned) - 1)}


# ==============================================================================
# 一、三个评估指标（从零规则版）
# ==============================================================================
def faithfulness(answer: str, contexts: list[str]) -> float:
    """忠实度：答案内容 bigram 在【检索到的上下文】中的覆盖率 ∈ [0,1]。

    高=答案有据（每句话都能在上下文里找到支撑）→ 不幻觉；
    低=答案编了上下文里没有的东西 → 幻觉。
    真实版（ragas）：让 LLM 把答案拆成陈述句，逐句判「能否由上下文推出」。
    """
    a = _bigrams(answer)
    if not a:
        return 1.0
    ctx = _bigrams(" ".join(contexts))
    return len(a & ctx) / len(a)


def relevancy(answer: str, query: str) -> float:
    """相关度：query 内容 bigram 在【答案】中的覆盖率 ∈ [0,1]。

    高=答案切题；低=答非所问。
    """
    q = _bigrams(query)
    if not q:
        return 1.0
    ans = _bigrams(answer)
    return len(q & ans) / len(q) if ans else 0.0


def context_relevance(query: str, contexts: list[str]) -> float:
    """检索相关性：query 内容 bigram 在【召回上下文】中的覆盖率 ∈ [0,1]。

    高=检索召回了与问题相关的内容；低=检索没找对（该查的没查到）。
    对超纲问题，这个值低是【正常】的（库里本来就没有）。
    """
    q = _bigrams(query)
    if not q:
        return 1.0
    ctx = _bigrams(" ".join(contexts))
    return len(q & ctx) / len(q)


# ==============================================================================
# 二、把基线 RAG 跑一条 query，拿到 (答案, 召回上下文)
# ==============================================================================
def run_rag(store, query: str, top_k: int = 3):
    hits = retrieve(store, query, top_k=top_k)
    contexts = [c for c, _, _ in hits]
    prompt = build_prompt(query, contexts)
    answer = generate_dummy(prompt, hits)
    return answer, contexts


# ==============================================================================
# 三、Eval Harness：一批用例 → 逐条打分 → 聚合报告
# ==============================================================================
def evaluate(store) -> None:
    print("\n" + "=" * 72)
    print(" ① 批量评估：对一组用例逐条打分")
    print("=" * 72)
    cases: list[tuple[str, str]] = [
        ("RAG 的全称是什么？", "in_scope"),
        ("overlap 重叠的作用是什么？", "in_scope"),
        ("向量检索用什么距离度量？", "in_scope"),
        ("今天晚饭吃什么？", "out_of_scope"),  # 超纲：库里没有，应拒答
    ]

    header = f"{'问题':<22}{'类型':<12}{'忠实度':>7}{'相关度':>7}{'检索相关':>9}"
    print(header)
    print("-" * len(header.encode("gbk", "ignore")))  # 粗略分隔线
    in_scope_f, in_scope_r = [], []
    for q, kind in cases:
        answer, contexts = run_rag(store, q)
        f = faithfulness(answer, contexts)
        r = relevancy(answer, q)
        cr = context_relevance(q, contexts)
        tag = "✅" if kind == "in_scope" else "⚠超纲"
        print(f"{q[:20]:<20}  {tag:<8}{f:>8.2f}{r:>8.2f}{cr:>10.2f}")
        if kind == "in_scope":
            in_scope_f.append(f)
            in_scope_r.append(r)

    print("\n  [聚合] in_scope 用例平均："
          f"忠实度={sum(in_scope_f)/len(in_scope_f):.2f}  "
          f"相关度={sum(in_scope_r)/len(in_scope_r):.2f}")
    print("  → 正常 RAG：in_scope 忠实度/相关度高；超纲问题检索相关度低、RAG 拒答（防幻觉生效）。")


def demo_hallucination_contrast(store) -> None:
    """演示②：同一问题，'有据答案' vs '幻觉答案'，看忠实度如何区分。"""
    print("\n" + "=" * 72)
    print(" ② 忠实度区分幻觉：同一问题，有据 vs 编造")
    print("=" * 72)
    query = "RAG 的全称是什么？"
    _, contexts = run_rag(store, query)

    grounded = "RAG 的全称是 Retrieval-Augmented Generation，即检索增强生成。"
    hallucinated = "RAG 的全称是 Really Awesome Gadget，是一种游戏机。"

    print(f"  问题: {query}")
    print(f"  有据答案 : {grounded}")
    print(f"             忠实度 = {faithfulness(grounded, contexts):.2f}  （每句都有上下文支撑）")
    print(f"  幻觉答案 : {hallucinated}")
    print(f"             忠实度 = {faithfulness(hallucinated, contexts):.2f}  （编了上下文没有的内容）")
    print("  → 忠实度正是用来【自动捕捉幻觉】的指标（生产里用 LLM-as-judge 更准）。")


def main() -> None:
    print("=" * 72)
    print(" Mini RAG · 评估器（从零）—— 给 mini_rag 打分")
    print("=" * 72)
    store = build_index(Path(__file__).resolve().parent / "sample.txt", chunk_sents=3, overlap=1)
    evaluate(store)
    demo_hallucination_contrast(store)
    print("\n" + "=" * 72)
    print(" 评估跑通 ✓  → faithfulness(防幻觉) / relevancy(切题) / context_relevance(检索质量)")
    print(" 真实版：ragas 用 LLM-as-judge 做语义级判定（本文件规则版是 toy proxy）。")
    print("=" * 72)


if __name__ == "__main__":
    main()
